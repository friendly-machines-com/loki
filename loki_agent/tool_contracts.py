"""Model-facing tool definitions shared without importing execution backends."""

# Each process may own two children, and only three descendant generations
# may exist. Thus one root process can have at most 2 + 4 + 8 active
# descendants instead of allowing recursive Agent calls to grow without bound.
MAX_ACTIVE_DIRECT_SUBAGENTS = 2
MAX_SUBAGENT_DEPTH = 3

READ_DEFAULT_LINES = 2000
BASH_DEFAULT_TIMEOUT_MS = 300000
BASH_MAX_TIMEOUT_MS = 600000
BASH_MAX_OUTPUT_CHARS = 10_000_000
JOB_TAIL_CHARS = 20_000

# Ask is the one tool whose response comes from a person, not a process or
# the network. The bounds below are its public contract, stated here once
# and reused by the tool schema and the ACP front's validation.
ASK_TOOL_NAME = "Ask"
ASK_MIN_OPTIONS = 2
ASK_MAX_OPTIONS = 8
ASK_MAX_QUESTION_CHARS = 4000
ASK_MAX_LABEL_CHARS = 200
ASK_MAX_DESCRIPTION_CHARS = 1000


def without_ask_tool(tools):
    """Tool definitions with the question tool removed.

    The Ask tool is advertised exactly when the caller can put a question
    to a user; seam-less callers (headless, subagents) must not offer a
    tool that would always fail.
    """
    return [
        tool for tool in tools
        if tool["function"]["name"] != ASK_TOOL_NAME]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "\n".join([
                "Reads a file from the local filesystem.",
                "",
                "- `file_path` may be absolute or relative to the current Loki cwd.",
                f"- Reads up to {READ_DEFAULT_LINES} lines by default.",
                "- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters",
                "- Results are returned using cat -n format, with line numbers starting at 1",
                "- Reads images (PNG, JPG/JPEG, GIF, WEBP) and presents them visually.",
                "- Reading a directory, a missing file, or an empty file returns an error or system reminder rather than content.",
                "- Do NOT re-read a file you just edited to verify -- Edit/Write would have errored if the change failed, and the harness tracks file state for you.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Plain absolute or relative filesystem path "
                            "passed to file APIs. Do not use Markdown links "
                            "or URLs."),
                    },
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
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Plain absolute or relative filesystem path "
                            "passed to file APIs. Do not use Markdown links "
                            "or URLs."),
                    },
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
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Plain absolute or relative filesystem path "
                            "passed to file APIs. Do not use Markdown links "
                            "or URLs."),
                    },
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
                "- Commands run in the current Loki cwd. Use /cd DIR to change that cwd; cd inside a Bash command only affects that one command.",
                "- IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience for the user.",
                f"- `timeout` is in milliseconds: default {BASH_DEFAULT_TIMEOUT_MS}, max {BASH_MAX_TIMEOUT_MS}.",
                "- `run_in_background` starts a detached job with stdout/stderr spooled to files. Use Jobs/JobStatus/JobStop to inspect or control it. No `&` needed.",
                "",
                "# Git",
                "- Interactive flags (`-i`, e.g. `git rebase -i`, `git add -i`) are not supported in this environment.",
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
                        '- ls -> "List files in current Loki cwd"',
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
            "description": "Fast file pattern matching. Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\". Returns matching file paths sorted by modification time. Hidden paths and paths excluded by ignore rules (such as .gitignore) are skipped by default. Set hidden=true and no_ignore=true independently to override these exclusions; hidden, ignored directories may require both, even when explicitly named in the pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The glob pattern to match files against"},
                    "path": {
                        "type": "string",
                        "description": (
                            "Plain filesystem directory path. Defaults to "
                            "the current Loki cwd. Omit it for the default; "
                            "do not pass null, undefined, Markdown, or a URL."),
                    },
                    "hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories. Does not bypass ignore rules. Defaults to false.",
                        "default": False,
                    },
                    "no_ignore": {
                        "type": "boolean",
                        "description": "Bypass ignore rules, including .gitignore and .ignore. Does not include hidden paths by itself. Defaults to false.",
                        "default": False,
                    },
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
                    "path": {
                        "type": "string",
                        "description": (
                            "Plain filesystem file or directory path for "
                            "rg. Defaults to the current Loki cwd; do not "
                            "use Markdown links or URLs."),
                    },
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
                (
                    "- Explore: Read-only search agent for broad fan-out "
                    "searches - when answering means sweeping many files, "
                    "directories, or naming conventions and you only need "
                    "the conclusion, not the file dumps. It reads excerpts "
                    "rather than whole files, so it locates code; it doesn't "
                    "review or audit it. It may delegate further read-only "
                    "Explore searches. Specify search breadth: \"medium\" "
                    "for moderate exploration, \"very thorough\" for "
                    "multiple locations and naming conventions. (Tools: "
                    "Agent, Glob, Grep, Read, Skill, Jobs, JobStatus, TodoRead, "
                    "WebFetch, WebSearch)"
                ),
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
                "- Available skills are listed below in this tool's description",
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
                "- Use the session's current-date instruction when searching for recent information.",
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
    {
        "type": "function",
        "function": {
            "name": ASK_TOOL_NAME,
            "description": "\n".join([
                "Asks the user one multiple-choice question and waits for the answer.",
                "",
                "Use this when a decision, preference, or missing detail changes what you do next and you cannot determine it yourself. Do not use it to ask permission for work you can simply do, and do not ask what you can already infer from the conversation.",
                "",
                f"- Offer {ASK_MIN_OPTIONS}-{ASK_MAX_OPTIONS} concrete, mutually distinct options. The user can always answer with their own text instead of an option.",
                "- Set multi_select only when the choices are not mutually exclusive.",
                "- If you recommend an option, list it first and say why in its description.",
                "- The user may decline or dismiss the question. Either way the turn continues without an answer; neither response is consent.",
            ]),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", "minLength": 1,
                        "maxLength": ASK_MAX_QUESTION_CHARS,
                        "description": "The question to ask, phrased so it ends with a question mark"},
                    "options": {
                        "type": "array",
                        "minItems": ASK_MIN_OPTIONS,
                        "maxItems": ASK_MAX_OPTIONS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string", "minLength": 1, "maxLength": ASK_MAX_LABEL_CHARS,
                                    "description": "The answer text, shown as the option's name"},
                                "description": {
                                    "type": "string", "maxLength": ASK_MAX_DESCRIPTION_CHARS,
                                    "description": "One line explaining what choosing this option means"}
                            },
                            "required": ["label"]
                        },
                        "description": "The answer choices; at least two are required."
                    },
                    "multi_select": {
                        "type": "boolean", "default": False,
                        "description": "Allow the user to pick several options instead of exactly one"}
                },
                "required": ["question", "options"]
            }
        }
    },
]
