# Loki

Loki is a coding assistant you give tasks in ordinary language. It connects to a
large language model (LLM) that can inspect your project, edit files, and run commands.
You can use it in a terminal or an editor; both use the same agent.

**Its actions are real.** It does not ask for approval before every command or
file change. Keep backups or use version control, review its work, and run it in
an appropriately isolated environment. Your requests and project content can be
sent to the LLM service you select, which may charge for their use.

New to Loki? Follow [Start here](#start-here) through your first task. You do not
need a settings file, hooks, or knowledge of model protocols to get started.

**Guide:** [Everyday use](#everyday-use) | [Use an editor](#use-loki-in-an-editor)

**Reference:** [Connections and models](#connections-and-models) |
[Settings](#settings) | [Compaction](#context-compaction) | [Hooks](#hooks) |
[Skills](#skills) | [Status and accounts](#status-and-account-controls) |
[Troubleshooting](#troubleshooting-and-diagnostics) |
[Developer reference](#developer-reference)

## Start here

The walkthrough uses a new demo directory, not your existing project. A new
folder is **not** a sandbox: on POSIX, use a disposable virtual machine for tool
isolation, and follow the steps inside it. Do not give that VM writable access to
your host's home directory or unrelated projects. Loki does not create the VM for
you. A suitably configured container is another option, but not as secure as a VM.

On Windows, Loki uses an AppContainer instead. Read
[Windows installation](#windows-installation), including its
[controlled process-directory rules](#windows-process-directory-and-history),
before continuing with the terminal walkthrough.

### 1. Get Loki

For the POSIX source checkout, you need:

- Python **3.11.10 or later**; if using Python 3.12, **3.12.4 or later**.
- **Bash** and **ripgrep** (`bash` and `rg`) installed and available on `PATH`.
- **Git** to download the source. No Python runtime packages need installing.
- An interactive terminal; Linux with foot is tested.

Install missing prerequisites using your operating system's installation tools.
The POSIX examples below use **Bash**; start `bash` first if you normally use a
different shell. Check the commands are available:

```bash
python3 --version
bash --version
rg --version
git --version
```

Download the source and remember the launcher location:

```bash
mkdir -p "$HOME/src"
git clone https://github.com/friendly-machines-com/loki.git "$HOME/src/loki"
LOKI="$HOME/src/loki/loki.py"
```

If you already have a checkout, skip the clone and set `LOKI` to its `loki.py`.
`LOKI` is just a shell variable used by this guide, not an application setting.
Keep this terminal open for the following steps. In a new terminal, set the
variable again or use the launcher's full path.

Of course, having the loki directory itself be writable by the remote host
would be a terrible idea.  Make it read-only.

### 2. Connect an AI account

Choose **one** of these paths. A ChatGPT subscription and an OpenAI API account
are different ways to pay for model use; having one does not imply API credit
for the other. Other providers and custom endpoints are covered in
[Connections and models](#connections-and-models).

#### Option A: ChatGPT subscription

If your ChatGPT account supports this connection, log in before starting Loki.

**In Bash:**

```bash
"$LOKI" auth login openai
```

**In PowerShell, after Windows installation:**

```powershell
& $LOKI auth login openai
```

Follow the URL and browser instructions printed by the command. On a headless
or remote machine, use the same command with `--device-code`; your ChatGPT
account or workspace must allow device-code login.

#### Option B: Provider API key

Obtain an API key from your provider and arrange any required billing. For
example, OpenAI API keys are managed at
[platform.openai.com/api-keys](https://platform.openai.com/api-keys).
This example supplies an OpenAI API key without putting its value in a shell
command or this repository.

**In Bash:**

```bash
read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
```

**In PowerShell:**

```powershell
$key = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $key).Password
Remove-Variable key
```

The variable must be set in the terminal that starts Loki. Do not paste the key
into the conversation or commit it to a file in your project.

### 3. Open a project and select a model

Create a demo directory **outside the Loki checkout**. The POSIX example starts
Loki there. On Windows, keep Loki's process cwd in the protected installation
and pass the project separately as `--shell-cwd`. Later, substitute your real
project for the demo directory.

**In Bash:**

```bash
mkdir -p "$HOME/projects/loki-demo"
cd "$HOME/projects/loki-demo"
"$LOKI"
```

**In PowerShell:**

```powershell
New-Item -ItemType Directory -Force "$HOME\projects\loki-demo" | Out-Null
& "$HOME\src\loki\dist\loki-setup\loki-setup.exe" --configure "$HOME\projects\loki-demo"
Set-Location "$HOME\src\loki\dist\loki"
& $LOKI --shell-cwd "$HOME\projects\loki-demo"
```

The PowerShell setup command configures **this project**, not the Loki
installation. Configure each new Windows workspace before opening it. The launch
must retain the protected process cwd; do not replace it with the project path.
The initial project selection owns history under that project's `.loki/chats`;
changing `/cd` later does not move it.

You are now typing **inside Loki**, not in your shell. Enter:

```text
/model
```

Choose your provider and then a model, following the prompts. For option A,
choose **OpenAI ChatGPT subscription**. For option B, choose the API provider
whose key you supplied, not the subscription connection. Loki may ask you to
confirm the service addresses to which it will send the credential; approve
only the service you intend to use.

Starting disconnected is normal until you make this selection. Check the
status area's model and working directory before asking it to work.

### 4. Try a first task

**Inside Loki**, enter this question and press Enter:

```text
What files are in this folder?
```

A message ending in `?` is handled as a read-only question for that turn. Loki
can inspect files but cannot use action tools such as Bash or file editing.

Next, give it a small instruction, **not a question**:

```text
Create hello.sh that prints "Hello from Loki", and run it with Bash.
```

Loki chooses the tools needed for the task, applies changes, and reports the
result. Read its response and open `hello.sh` in your editor to inspect what it
created. Do not assume an agent's success claim is sufficient verification.

Press **Ctrl+C** if you need to stop a running response. At the idle Loki prompt,
enter `/quit` to return to your shell. You can then check the result yourself:

```bash
bash hello.sh
```

In PowerShell, your shell remains in the protected launch directory, so use the
script's project pathname instead:

```powershell
bash "$HOME\projects\loki-demo\hello.sh"
```

It should print `Hello from Loki`. If it does not, reopen the conversation and
give Loki the actual error rather than starting over without that evidence.

In a Git project, review `git status --short` and `git diff` after a task. New,
untracked files appear in `git status` but not in ordinary `git diff`; open
those files too. The empty demo folder need not be a Git repository.

### 5. Return to the conversation

On POSIX, reopen the saved-chat picker from the **same project directory**.
On Windows, retain the controlled process cwd instead; pass the tool directory
as a parameter, as below.

**In Bash:**

```bash
"$LOKI" resume
```

**In PowerShell:**

```powershell
Set-Location "$HOME\src\loki\dist\loki"
& $LOKI --shell-cwd "$HOME\projects\loki-demo" --resume
```

The Windows command uses `--resume`, not the POSIX positional `resume` form.
Its picker reads the selected workspace's `.loki/chats`, independently of
Loki's process cwd. For an existing log elsewhere, use `-r` with its full
pathname; see [saved conversations](#saved-conversations).

Choose the conversation by number. Supply API-key variables again when starting
from a new shell; subscription login is stored separately. A saved connection
may require confirmation before use.

You now have the basic loop: **describe a task, watch it work, inspect the
changes, verify the result, and give feedback**. The remaining sections explain
controls and optional features; they are not prerequisites for a first task.

## Windows installation

Use Windows **11 24H2 (build 26100) or later**, or **Windows Server 2025 or
later**. The AppContainer setup applies to desktop applications.

The source-build route below requires Python **3.12.4 or later**, Git,
**ripgrep 15** (`rg.exe`), and **Bash** (`bash.exe`, for example from Git for
Windows). PowerShell and `cmd.exe` are not substitutes for the Bash executable
used by Loki's tools. Ensure these programs are on the launching terminal's
`PATH`; restart a terminal or editor after changing it.

**In PowerShell:**

```powershell
python --version
rg --version
bash --version
git --version
New-Item -ItemType Directory -Force "$HOME\src" | Out-Null
git clone https://github.com/friendly-machines-com/loki.git "$HOME\src\loki"
Set-Location "$HOME\src\loki"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "pyinstaller==6.16.0"
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --console --name loki loki.py
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --console --name loki-acp loki-acp
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean loki-setup.spec
$LOKI = "$HOME\src\loki\dist\loki\loki.exe"
Set-Location "$HOME\src\loki\dist\loki"
```

These are the executable bundles built by the Windows CI workflow. Stop and fix
any build error before continuing. Keep the complete `dist\loki`,
`dist\loki-acp`, and `dist\loki-setup` directories, including their supporting
files; copying just the `.exe` files is insufficient. If you already have these
built bundles, use their actual locations instead of rebuilding.

### Windows process directory and history

Loki's Windows process cwd is a fixed, trusted launch location, not a project
selector. Keep it in the protected deployment directory, outside locations the
agent can modify. This includes the terminal runtime, ACP front/worker and helper
processes. Windows relative-file, DLL and executable lookup must not acquire a
model-writable search location through a user or model project request.

The terminal's initial `--shell-cwd` selects the **workspace** and initial tool
directory. ACP selects its workspace at session admission. History belongs to
that workspace's `.loki/chats`, not to the deployment or process cwd. A later
`/cd` changes only the logical directory for tools; neither it nor a restored
chat's saved tool cwd relocates the workspace's history.

Ordinary tool subprocesses may run in the requested project directory without
moving Loki's agent process. The PowerShell examples keep `dist\loki` as the
controlled launch directory and pass the project with `--shell-cwd`. Do not make
the bundle writable or change the process cwd to select another project.

Return to [step 2: connect an AI account](#2-connect-an-ai-account), then configure
the workspace as shown in step 3.

To check an existing workspace or edit its grants, use its actual directory:

```powershell
& "$HOME\src\loki\dist\loki-setup\loki-setup.exe" --verify "$HOME\projects\loki-demo"
& "$HOME\src\loki\dist\loki-setup\loki-setup.exe" --edit "$HOME\projects\loki-demo"
```

`--configure` is noninteractive; `--edit` opens the permissions editor. Running
`loki-setup` without arguments does not configure the current project for you.
Keep workspaces separate from the Loki deployment, interpreter, credentials,
configuration, and state directories; writable grants overlapping protected
storage are refused.

Windows tool scratch files use `<workspace>\.loki\tmp` through `TEMP` and `TMP`.
Loki creates that directory as needed but does not clean it automatically; remove
unneeded contents between runs.

## Everyday use

### Asking, planning, and making changes

You do not need to invoke tools such as Glob, Grep, or Edit yourself. Tell Loki
the outcome you want. For example:

- "Find where configuration is loaded and explain it. Do not change files."
- "Plan a fix for this error before implementing anything."
- "Implement that plan, run the relevant tests, and show what changed."
- "Search the web for the library's documentation and check this usage."

Loki can search files and text, read and edit files, run Bash commands, search
and fetch web pages, delegate exploration to helper agents, and maintain a task
list. A Bash invocation is ephemeral: changes to its shell state do not become
persistent application state. Use `/cd` to change the working directory.

Questions and instructions are distinct. A message ending in `?`, or containing
the literal `what?` without regard to case, restricts the whole turn to
read-only tools. This includes refusing Bash even for a command such as
`git diff`. To request an action, give an instruction such as "Run git diff and
summarize it" rather than "Can you run git diff?". The next message gets a fresh
decision.

Loki validates tool input, repairs some unambiguous formatting mistakes, and
reports corrections. Other invalid calls are rejected. Neither these checks nor
a read-only mode replace operating-system isolation.

### Terminal controls and modes

- **Enter** submits the input.
- **Shift+Enter** inserts a newline when supported by the terminal. Pasting
  multiline text is another option.
- **Up/Down** browse earlier user inputs.
- **Ctrl+C** requests cancellation of the current work or input. In a startup
  picker it can cancel startup; it is not a universal "quit and undo" key.
- **`/quit`** exits from the idle prompt. Cancel running work first if needed.
- **Ctrl+D** on empty input ends the session.
- **Shift+Tab** cycles modes for the **next** turn; it does not change work
  already in progress. Watch the `mode` field in the status area.

Modes are:

- **`normal`:** answer or act according to the request.
- **`explore`:** inspect and answer with read-only tools, without editing project files.
- **`plan`:** inspect and produce a plan; updating the todo list is allowed.
- **`edit`:** implement requested changes and verify them.

The status area shows the selected service/model, context measurement, working
directory, current activity, and available commands. `/ps` is bold while jobs
are running; `/compact` is bold while compaction is executing.

### Working directory and commands

Slash commands are entered **inside Loki**. Command-line options such as
`resume` are passed when starting it **from your shell**. For startup options,
run `"$LOKI" --help` in Bash or `& $LOKI --help` in PowerShell.

- `/pwd` shows the directory used for relative file paths and Bash commands.
- `/cd DIR` changes that directory. It does not relocate the current chat's
  saved-session directory. On Windows the directory must be permitted by the
  configured container.
- `!command` runs a shell command directly, without asking the model to generate
  it. For example, `!bash hello.sh` runs the demo from the walkthrough.
  The result is sent to the model as well.
- `/model` selects a connection/model; switching does not discard the chat.
- `/todo` displays the current task list, including while the agent works.

The launcher's `--shell-cwd DIR` option sets the initial logical tool directory;
`/cd` updates that same kind of parameter. Neither changes Loki's OS process cwd.
Loki runtimes, workers and helpers retain the controlled process cwd, while an
ordinary tool subprocess can be launched with the requested project directory.
Keep the Windows launch rules separate from the POSIX walkthrough; see
[Windows process directory and history](#windows-process-directory-and-history).

Most successful Bash stdout, Read file bodies, and loaded Skill instructions
are hidden from automatic terminal display by default, but the model still
receives them. Errors, stderr and status notes remain visible. Use `!command`,
`/ps ID`, or [display settings](#tool-output-display) when you need the output
yourself. Ask Loki to summarize its evidence, not merely assert that it ran a
command.

### Images

Use `/image PATH` to stage a PNG, JPEG, GIF, or WebP image for the next prompt.
Relative paths use `/pwd`; quote paths containing spaces. Repeat the command to
stage several images, or send an empty prompt to submit the images alone.

Each image is limited to **20 MiB** and is snapshotted when staged, so later
changes to the file do not change what is sent. The selected model must support
image input. See [queued input](#queued-input) to inspect or remove staged
images.

The model can also use Read on a local image during a turn. That sends the
snapshot immediately as visual tool-result content, with the same formats and
size limit; no extra user prompt is needed. The terminal, editor, and replay
show a short notice while the model request and saved conversation retain the
image. Image type is detected from file signatures. Read's line offset/limit
apply only to text. Loki does not capture the terminal framebuffer or perform
local OCR, resizing, or pixel decoding.

### Background jobs

Ask for work to run in the background when appropriate, then inspect it with:

```text
/ps
/ps all
/ps ID
/ps stop ID
/ps kill ID
```

Replace `ID` with the job's identifier. `/ps` lists running, starting, and failed
jobs, oldest first; `/ps all` includes finished jobs. `/ps ID` shows status and
recent stdout/stderr. `stop` requests graceful termination; `kill` forces it.

These terminal controls work while a turn is running and during pickers or
confirmations. They do not wait behind a model response. `/ps` is not an ACP
command. Job listings report the working directory captured at launch.

### Queued input

The terminal can accept input while the agent is busy. Ordinary prompts wait in
send order; monitor commands can answer immediately.

```text
/queue
/queue texts
/queue images
/queue texts delete ID
/queue texts edit ID REPLACEMENT TEXT
/queue texts move ID before OTHER_ID
/queue texts move ID end
/queue images delete ID
/queue images move ID before OTHER_ID
/queue images move ID end
```

`txt` and `img` are aliases for `texts` and `images`. IDs identify entries, not
positions: they survive edits/moves, are not reused after removal, and differ
even for identical submissions. Both entries in a move must still be pending;
stale IDs cannot accidentally modify a replacement entry. Successful edits
re-list the queue. A staged image cannot be repointed to another file: delete it
and stage the new file with `/image`.

`/todo`, `/ps`, `/queue`, `/status` (including `--json`, `all`, and `save`), and
read-only `/account CONTROL` requests answer while a turn is running. Delayed
answers appear separately as they complete, without becoming model input.
Interactive `/account` and `/account CONTROL ACTION` requests wait for the turn
to finish. In ACP, wait for the current prompt before issuing these commands;
editor input queueing also depends on the client.

### Saved conversations

Both frontends save and list chats under the selected workspace's `.loki/chats`.
The terminal captures that workspace from its initial `--shell-cwd` argument, or
its initial directory when the option is absent. ACP uses the workspace supplied
when opening the session. Later `/cd` changes and restored tool directories do
not relocate history. On Windows, select the project with `--shell-cwd` while
retaining the [controlled process cwd](#windows-process-directory-and-history).

Logs retain the selected model, connection settings and other session state,
**not credential values**. They can still contain sensitive project data.

On POSIX, use `"$LOKI" resume` from the original startup directory for the picker,
or `resume LOG` for a particular chat. On Windows, retain the protected launch
directory and use `--resume` for the picker, as in the walkthrough; positional
`resume` is rejected by its launch gate.

If you selected a workspace with `--shell-cwd`, supply it again for the picker
or saved-ID lookup. For a known existing log, use `-r LOG` (or `--resume=LOG`).
Here `LOG` must be replaced with the actual pathname, not typed literally. For example, in
PowerShell, substitute the actual log filename for `chat-ID.json`:

```powershell
& $LOKI --shell-cwd "$HOME\projects\loki-demo" -r "$HOME\projects\loki-demo\.loki\chats\chat-ID.json"
```

This does not change the process cwd. Quote pathnames containing spaces.

Supply the connection's credentials again when necessary. Loki asks before
using saved endpoints; missing credentials do not erase the saved connection.
Changing `/model` preserves the conversation, though provider-specific
reasoning data may not transfer between models.

`LOKI_*` connection settings initialize new sessions. On resume they temporarily
override the saved connection; choosing a model with `/model` updates the saved
choice. Older savefile formats are not automatically migrated. Current Loki
loads schema v4 and compacted schema v5 logs; [compaction](#context-compaction)
does not delete the archived history.

## Use Loki in an editor

An ACP client is an editor component that starts Loki and exchanges tasks and
results with it. You normally configure **the `loki-acp` launcher**, not
`loki.py`. Model selection, tools, saved chats and compaction are the same agent
features, presented through the editor's interface.

Run the editor and Loki in the same intended, isolated environment as the
terminal walkthrough. A later `export` in another shell does not update an
already-running editor. Supply API-key environment variables before starting
the editor/agent, or log in with the same user configuration directory for a
subscription connection. Restart the ACP agent after changing startup settings.

On Windows, the client must launch `loki-acp.exe` from a fixed, protected process
directory, not from the project's writable directory. Pass the project as ACP
session data instead. An absolute executable pathname alone does not fix its
process cwd. A client configuration must preserve this distinction; do not infer
a safe Windows setup by replacing the executable in a POSIX example.

### Emacs with agent-shell

Install [agent-shell](https://github.com/xenodium/agent-shell#installation) and
its dependencies, then add a custom agent configuration. This POSIX example
uses the agent-shell/acp.el APIs present in this checkout and assumes the default
list-valued `agent-shell-agent-configs`:

```elisp
(require 'agent-shell)

(add-to-list
 'agent-shell-agent-configs
 (agent-shell-make-agent-config
  :identifier 'loki
  :mode-line-name "Loki"
  :buffer-name "Loki"
  :shell-prompt "Loki> "
  :shell-prompt-regexp "Loki> "
  :client-maker
  (lambda (buffer)
    (acp-make-client
     :command "/home/YOU/src/loki/loki-acp"
     :context-buffer buffer
     :environment-variables
     (agent-shell-make-environment-variables :inherit-env t)))))
```

Replace the command with the **absolute path** to your POSIX checkout's
`loki-acp`. Do not leave `YOU` or use an unexpanded `~`. This snippet does not
establish the separately controlled server process cwd required on Windows.
If you customize agent-shell's configuration through a function, return this
configuration from that function rather than applying `add-to-list` to it.

Open a file in the project and run **`M-x agent-shell`**, then select Loki. Use
**`C-u M-x agent-shell`** to force a new shell. Select a model through the
client's controls or `/model`, then give it an ordinary task.

This is a direct local-client example; it does not apply agent-shell command
wrappers for SSH, containers, or other remote execution. Configure those through
the client's documented facilities rather than assuming this example provides
isolation. Other ACP editors similarly need the executable path, project
workspace, and startup environment; follow that client's configuration format.

The agent-shell version in this checkout requests ACP **v1**. Loki supporting
v2 does not automatically upgrade a client that requests v1. Wire differences
are described in [ACP protocol notes](#acp-protocol-notes), not needed to send
your first editor task.

## Connections and models

A **provider** is the service running the AI model; a **model** is the particular
AI you choose there. The `/model` picker uses models.dev and offers providers
for credentials captured at startup, such as `OPENROUTER_API_KEY`. Deprecated
models remain selectable but are labelled.

Before sending a catalog credential, Loki asks whether it may be sent to the
specific request URLs, including model-list alternatives and applicable
account-control URLs. Approval is for complete URLs, not an entire host.
Approvals are saved in the protected credential database; only the supervisor
reads/writes it. Explicit launch settings and Loki's canonical subscription
routes need no saved endpoint approval. Saved-chat confirmation is separate.

Removing an approval affects future credential requests, not a credential
already issued. The old state-directory `provider-endpoints.json` is not loaded
or imported.

ChatGPT subscription credentials are not sent to models.dev or OpenAI's API
billing service. The subscription connection lists models available to that
account; those may differ from API-billed models.

### Custom endpoints and startup overrides

For an explicitly configured OpenAI- or Anthropic-compatible endpoint, provide
its actual URL, model identifier and credentials. This Bash example selects an
OpenCode endpoint; it is an **alternative** to the account-picker walkthrough,
not something to combine with it:

```bash
export LOKI_API_BASE="https://opencode.ai/inference/go/openai/v1/chat/completions"
export LOKI_MODEL="glm-5.2"
read -r -s -p 'Endpoint API key: ' LOKI_API_KEY
printf '\n'
export LOKI_API_KEY
"$LOKI"
```

Obtain the key from the service whose URL you configure. Its model availability
can change. The older base
`https://opencode.ai/zen/go/v1/chat/completions` remains supported.

Without an explicit `LOKI_API_BASE` or a resumed connection, Loki starts
unconnected; choose one with `/model`. It does not silently choose a provider
or the first model returned by a server. An explicit endpoint needs `LOKI_MODEL`
or a model choice before the first chat request.

The explicit endpoint uses `LOKI_API_KEY` if supplied; otherwise it sends no
authentication header. Loki does not borrow another provider's credential merely
because the wire protocol matches. A complete captured connection also appears
as `Explicit LOKI_* connection` in `/model`, including when the catalog is
unavailable or after you switch to another provider.

Connection variables are `LOKI_API_BASE`, `LOKI_PROVIDER`, `LOKI_MODEL`,
`LOKI_API_KEY`, `LOKI_MODELS_URL`, `LOKI_MAX_TOKENS`, `LOKI_CONTEXT_WINDOW`,
`LOKI_AUTH_HEADER`, `LOKI_AUTH_SCHEME`, `LOKI_ANTHROPIC_VERSION`, and `LOKI_STREAM`.
`LOKI_PROMPT_CACHE` controls Anthropic Messages caching. Set them before launch;
changing an unrelated shell's environment cannot change a running session.
Clear old explicit endpoint/model overrides if you want to return to the
account-picker startup flow.

Direct Anthropic API connections enable prompt caching by default. For other
Anthropic-compatible services use `LOKI_PROMPT_CACHE=1`; set it to `0` to disable
caching. Provider cache reuse can depend on the conversation and directory.

### Thinking and trace display

`/thinking` shows and changes the selected model's supported settings:

```text
/thinking effort high
/thinking mode off
/thinking mode adaptive
/thinking mode manual budget 2048
/thinking retention preserve
```

Not every model supports every setting. Manual thinking needs an output limit
larger than the allowance. Retention asks a supported provider to reuse earlier
thinking and may increase input tokens and cost. Use `default` in place of a
value to clear a preference.

Preferences are remembered with the conversation. Inapplicable settings become
inactive when switching models and return if you switch back. Changes during a
response apply to the next turn. ACP clients expose the same settings; use
commands for numerical allowances or changes that must happen together.

`/trace thinking on` displays thinking/summaries returned by the provider;
`/trace thinking off` hides them. These commands do **not** enable thinking or
change how much the model does. Hidden thinking and continuation data remain in
history. Some models return no readable thinking; OpenAI exposes summaries,
not its complete internal reasoning.

### Streaming and context usage

`LOKI_STREAM=1` requests streamed assistant display. Streaming is off by default;
if the server rejects it, use `LOKI_STREAM=0`. Provider-specific connections can
have their own streaming requirements.

The status area's `Context: 37%` is the **last reported** context usage, not a
precise estimate of your next prompt. `*` marks an older/incomplete measurement;
`unknown` means usage or capacity is unavailable. `LOKI_CONTEXT_WINDOW` sets a
positive capacity override for this display, not a larger limit at the service.
For long conversations, see [context compaction](#context-compaction).

## Settings

Most users can use Loki without a configuration file. When needed, edit the
optional user `settings.ini`:

- POSIX: `~/.config/loki/settings.ini`.
- Windows: normally `%LOCALAPPDATA%\loki\settings.ini`.
- A nonempty `XDG_CONFIG_HOME` overrides the base on both platforms:
  `$XDG_CONFIG_HOME/loki/settings.ini`.

Startup does not create or rewrite the file. Hooks are loaded from this user
configuration, **not from a repository's settings**. Restart Loki or your ACP
agent after editing it.

### Tool-output display

To show successful tool bodies that are hidden by default, add:

```ini
[terminal]
show_bash_stdout = true
show_read_stdout = true
show_skill_stdout = true
```

Stderr, errors, status and tool notes remain visible without these options. The
settings also affect automatic shell-job status reports, but not explicit
inspection using `!command` or `/ps ID`, and do not change ACP output.

Hidden output is still captured, saved and sent to the model: these are display
preferences, not redaction. New saved results retain stream identity for replay.
Older combined Bash output cannot be reliably split and shows a notice when
stdout is hidden; enable display to see it in full. Older Read results similarly
show a notice when hidden. New Read results keep errors, empty/binary/image
notices and tool notes visible in either view.

Preference updates preserve hook sections, their order and unknown values, but
rewrite formatting and do not retain comments. Keep disabled hook definitions
somewhere safe rather than relying on commented-out settings surviving updates.
Unknown sections/options are reported but preserved; unknown hook phases do not
run. Malformed supported configuration fails startup rather than silently
removing hooks.

### Custom system instructions

Create `SYSTEM.md` in the same user configuration directory as `settings.ini` to
replace the main conversation's built-in base instructions. To inspect the
current default, run this **from your shell**, not inside Loki:

```bash
"$LOKI" --print-default-system-prompt
```

Save and edit the output if you want a starting point. This export needs no
provider, credentials or runtime setup. The file is literal UTF-8, not a
template; an empty file omits the base instructions. Date/directory information,
tool definitions and mode restrictions are added separately.

Changes apply to **new conversations**, not current or resumed chats: saved
chats retain their original instructions. Deleting the file restores the
built-in default for the next new chat. Loki never creates or rewrites this
file on startup, nor automatically upgrades your custom instructions.

If reading/decoding fails, Loki reports the problem and uses the default without
changing your file. Explore helpers, headless Explore runs and WebFetch's
analysis helper keep their own specialized instructions instead.

### Credentials and isolation

Use the account setup in the walkthrough; for more login controls, these Bash
examples run outside Loki:

```bash
"$LOKI" auth status openai
"$LOKI" auth logout openai
"$LOKI" auth login openai --device-code
```

In PowerShell, use `& $LOKI` for the executable. Browser login tries to open a URL
and needs localhost port **1455 or 1457** for confirmation. Add `--no-browser` to
print the URL without launching a browser; that still requires the callback to
reach the machine running Loki. Device-code login avoids that local callback
and is useful for remote/headless machines when the account permits it. An
interrupted credential refresh may require logging in again.

Subscription tokens are stored in the user configuration directory's
`credentials/tokens.json`, normally `~/.config/loki/credentials/tokens.json` on
POSIX. No desktop credential service or message bus is required. Logout removes
the local login but does not immediately revoke credentials already issued to
running sessions; close those sessions too.

Loki captures credentials at startup, then removes variables ending in `_KEY`,
`_TOKEN`, or `_PAT` from environments passed to tools and hooks. On Linux it also
hides the credential directory from tools/subagents. These protections are not a
general-purpose tool sandbox: POSIX users still need their VM/container, and
Windows users need the configured AppContainer.

On Linux, credential isolation requires permitted user/mount namespaces and
`/proc`; startup fails if they are unavailable. A locked-down container can
therefore fail before displaying a prompt. Other POSIX systems do not gain the
Linux credential-mount cover. Avoid weakening your host's security settings just
to make an unknown container configuration start; use an environment you control.

## Context compaction

A model has a limit on the amount of conversation it can consider at once -- its
**context**. Long chats can approach that limit even though all their history
still exists on disk. Compaction makes a smaller active context so you can
continue the conversation; it does not delete the saved transcript.

**Inside Loki**, request it with:

```text
/compact
```

Actual remote compaction requires a saved conversation using the **native OpenAI
ChatGPT subscription Responses connection**, including its selected Responses-Lite
profile. It is not a general feature of every OpenAI-compatible API. Loki does
not silently change your service, model or profile to compact.

- During a turn, compaction waits for the current model sampling and complete
  tool batch to finish, then that turn continues using the new context.
- Repeated requests while waiting/running share one operation.
- In an idle chat, the command compacts without starting another coding task.
- If there is nothing removable, such as startup instructions alone, it reports
  `Nothing to compact` without a hook, save or provider request.
- There is no automatic threshold, overflow or model-change trigger. Compact
  **before** the service rejects an oversized request; compaction itself needs
  to send context and is not a guaranteed rescue after overflow.

The current retention policy keeps saved system/developer instructions and user
messages, followed by a new checkpoint. Older assistant/tool history remains in
the archive but is not supplied verbatim to later inference. The checkpoint is
lossy: something saved in an old tool result is not automatically remembered or
retrieved later. Chats dominated by user messages can remain large.

Compaction reduces the prompt, **not disk or RAM use**: the full JSON archive is
still saved and held in memory. Context usage becomes unknown after replacement
until an ordinary response reports another measurement.

### Cancellation, failures, and editors

Cancel ongoing work with Ctrl+C in the terminal or the editor's cancellation
control. A rejected policy, failed request or cancellation before publication
leaves the old context selected. Once the new checkpoint is published, a later
cleanup error is reported without undoing that checkpoint.

During an active turn, actual compaction-request failures use that turn's
configured [retry hook](#retrying-model-requests). Local admission/policy/save
errors are not model-request retries. An unrecovered mid-turn failure ends the
turn while preserving completed work and attempting its normal notification.
Idle failures leave the frontend usable; idle/post-turn compaction does not
invoke the turn-error hook.

A client that queues input while busy may not deliver `/compact` until the turn
ends; server support alone cannot bypass the client's queue. Loki supports
negotiated ACP v1 and v2, with different acknowledgement/progress behavior;
see [ACP protocol notes](#acp-protocol-notes) for integrations. The trusted
[pre-compaction hook](#pre-compaction-policy) can permit or deny an attempt.

## Hooks

Hooks are your programs that Loki runs at particular points. They can check or
adjust tool calls, format files, notify you when a turn ends, or choose whether
to retry a failed model request. They are optional. Loki includes a retry
program; the other programs are ones you supply.

Add sections to the user `settings.ini` described under [Settings](#settings).
Hooks of the same kind run in file order. Restart Loki or the ACP agent after
changing them; remove a section and restart to disable that hook.

### A working notification example

Save this as a trusted file such as `/home/me/bin/loki-notify.py`:

```python
import json
import sys

json.load(sys.stdin)
print("Loki finished a turn", file=sys.stderr)
```

Add this section, replacing **all three paths** with your Python executable,
script, and an existing directory:

```ini
[hook.turn_end.notify]
command.0 = /absolute/path/to/python3
command.1 = /home/me/bin/loki-notify.py
cwd = /home/me
# Optional execution limit, in milliseconds:
# timeout_ms = 5000
```

The hook consumes its input, exits successfully with empty stdout, and reports
its message on stderr. After restarting, send a prompt; the message should
appear when its turn finishes. Replace the stderr action with your own
notification action when ready. This hook runs for success, failure and
cancellation in the terminal and ACP, not for local commands/internal subagents.
A notification error cannot change the chat result.

On Windows, use actual Windows paths, including the interpreter's `.exe`.
Find the Python executable you intend to use with
`python -c "import sys; print(sys.executable)"` in that interpreter's environment.

### Command configuration

Every hook requires `cwd`: an existing working directory independent of the
chat's directory. Use an absolute path. Put the executable in `command.0` and
each argument in the next numbered option, with no gaps.

INI quotes are literal, not shell quoting: arguments containing spaces do not
need quotes around them. Leading/trailing whitespace is trimmed, `%` is literal,
and environment-variable names in values are not shell-expanded.

Programs receive one JSON object on stdin followed by EOF. Use stdout for a
single JSON reply object, **not logs**; empty successful stdout is also accepted.
Use stderr for logs. Nonzero exit, timeout, malformed/non-object JSON or output
beyond the hook limits is an error. There is **no timeout by default**; configure
`timeout_ms` if you want a limit. A hook with no limit can hold up the operation.
Hooks never receive Loki's API credentials, but their input/log access can still
expose conversation or project content.

### Tool checks and formatters

Tool hooks receive an `event` identifying the phase and an `invocation` object.
The useful starting fields are `tool_name`, `effective_arguments`, and `cwd`.
Other fields include `call_id`, `raw_arguments`, `original_arguments`, `schema`,
`model`, `provider`, `parse_error`, `validation_issues`, and `adjustments`.

- **`pre_tool_call`:** inspect or replace arguments before execution. Reply with
  `action: "continue"` (the default) or `"deny"`; optional `arguments` replaces
  the proposed arguments. `message`/`note` can explain the decision.
- **`pre_tool_gate`:** approve/deny after arguments pass validation. The same
  actions apply, but returning `arguments` is forbidden.
- **`post_tool_call`:** inspect the result, for example to format files. Input
  adds `outcome` with `status`, `executed`, `ok` and `content`. It can append a
  `note`, not replace/veto the executed result. It also runs after rejected or
  invalid calls, so check the outcome rather than assuming execution occurred.

Here is a complete gate that denies Bash calls. Save it as your own trusted
`deny-bash.py`:

```python
import json
import sys

request = json.load(sys.stdin)
if request["invocation"]["tool_name"] == "Bash":
    print(json.dumps({"action": "deny", "message": "Bash is disabled by this hook."}))
```

Register it with actual paths:

```ini
[hook.pre_tool_gate.no_shell]
tools.0 = Bash
command.0 = /absolute/path/to/python3
command.1 = /home/me/bin/deny-bash.py
cwd = /home/me/project
```

This demonstrates the hook API; it is not a sandbox and does not disable other
tools. Omit `tools.0`, `tools.1`, etc. to apply a tool hook to all tools. For
argument adjustments or formatting, configure your corresponding programs, for
example:

```ini
[hook.pre_tool_call.normalize]
tools.0 = Write
tools.1 = Edit
command.0 = /home/me/bin/loki-normalize
cwd = /home/me/project
on_error = deny

[hook.post_tool_call.format]
tools.0 = Write
tools.1 = Edit
command.0 = /home/me/bin/loki-format
cwd = /home/me/project
on_error = continue
workspace_side_effects = yes
```

`loki-normalize` and `loki-format` here are **your programs**, not bundled
executables. Write/install them before enabling those example sections. By
default, a before-tool hook error blocks the call; a post-tool hook error is
reported but cannot undo work. `on_error = continue` is available
for pre-tool hooks if you explicitly want that behavior; post-tool hooks cannot
use `deny`.

For `post_tool_call`, image Read results in `outcome.content` include text and an
image block with base64 `source` fields `type`, `media_type`, and `data`. A text
note does not replace the image with text. `outcome.content` can otherwise be a
text string or a canonical content-block list.

If your tool hook modifies project files, declare `workspace_side_effects` and
report `changed_paths` where possible so Loki invalidates stale file
observations. Paths use the invocation's working directory. Declared effects
without a nonempty path list invalidate all observations; they do not undo
filesystem changes. Do not treat hooks as transactions around arbitrary tools.

### Pre-compaction policy

After writing a trusted policy program, register it using its real path:

```ini
[hook.pre_compact.policy]
command.0 = /absolute/path/to/python3
command.1 = /home/me/bin/precompact-policy.py
cwd = /home/me/project
```

Input fields are `event` (`pre_compact`), `operation` (`remote_v2`),
`conversation_id`, `chat_log_path`, `chat_log_cwd`, `model`, and `provider`.
`chat_log_cwd` is the lookup context for the log pathname, not necessarily the
hook's configured `cwd`. The log can contain sensitive history.

Empty successful stdout or `{"action":"continue"}` permits the attempt;
`{"action":"deny","message":"Not now"}` rejects it. The reply accepts only
`action`, `message`, and `changed_paths`, not replacement instructions/context.
Errors abort the operation; `pre_compact` cannot use `on_error = continue`.
The hook runs once per compaction operation, not once per model-request retry.

If declared, pre-compaction workspace effects invalidate file observations
before the program starts, including on failure/cancellation. They cannot be
rolled back. Reported changes also invalidate observations. Turn-end and
turn-error hooks do not accept `workspace_side_effects`.

### Retrying model requests

Enable the bundled retry hook to keep a prompt active through provider/network
outages rather than resubmitting it yourself. This works in the terminal and
ACP, and for compaction requests made during an active turn.

Add this section, replacing the executable, checkout, and directory paths:

```ini
[hook.turn_error.retry]
command.0 = /absolute/path/to/python3
command.1 = /absolute/path/to/loki/contrib/hooks/retries.py
cwd = /an/existing/directory

# Optional: uncomment both lines to allow at most six retries per prompt.
# command.2 = --max-retries
# command.3 = 6
```

After restarting Loki/the ACP agent, the defaults are:

- Ordinary failures wait **5 seconds** before the first retry.
- Recognized server overload waits **10 minutes** before the first retry.
- Waits double, up to **30 minutes**.
- There is **no retry-count limit** unless `--max-retries` is supplied.
- Recognized context-limit failures stop; retrying does not shrink or
  automatically compact the conversation.

Pass `--initial-delay`, `--overload-initial-delay`, `--multiplier`, or
`--max-delay` as further numbered arguments, with each option and value in
separate entries. Delays are seconds; the multiplier is a growth factor. A
provider's requested retry wait can exceed the selected cap. Run the script's
`--help` for all options.

Loki reports the failure reason and next wait. In ACP, inspect agent/server logs.
Ctrl+C (terminal) or cancellation (editor) stops waiting. Waiting cannot repair
invalid credentials; some quotas require an account reset or limit change.
Retries can repeat generated output/provider work and can be billed again.
This hook retries **model requests**, not failed Bash commands or other tools.

For your own recovery program, input includes `event`, `retry_count`,
`error_type`, `error_message`, `recovery_point`, `output_started`,
`response_saved`, and `cwd`. Recovery points are `request_replay` or
`history_resample`. Additional provider/network diagnostics can include
`provider_code`, `provider_error_type`, `error_category`, `http_status`,
`network_errno`, `error_details`, `rate_limit_headers`, `incomplete_reason`, and
`retry_after_seconds`.

Reply with `{"action":"finish"}` to stop recovery, or
`{"action":"retry","delay_seconds":5}` to retry. A delay must be finite,
nonnegative and not boolean. Empty successful output leaves the decision to
subsequent hooks; the first explicit decision wins. If none decides, or a hook
fails, recovery finishes with the original failure. Turn hooks do not take an
`on_error` setting. A broken retry hook stops recovery instead of retrying
itself: check its paths, Python executable, working directory and stderr.

To disable retries, remove the `hook.turn_error.retry` section and restart.
For a custom turn-end notifier, input contains `event`, `reason`, `cwd`, and
`text`; reasons include `completed`, `error`, `cancelled`, and `max_loops`, with
error diagnostics when available. Its returned object cannot change the result.

## Skills

A skill is reusable instructions for the model, stored in a Markdown file. Use
one for a repeatable workflow such as reviewing changes. It is not an executable
plugin, does not grant permissions, and is not a replacement for a model
connection.

### Install or create a skill

Copy its directory, including `SKILL.md` and supporting files, into:

- POSIX: `~/.config/loki/skills`.
- Windows: the LocalAppData known folder's `loki\skills`, normally
  `%LOCALAPPDATA%\loki\skills`.
- With `XDG_CONFIG_HOME`: `$XDG_CONFIG_HOME/loki/skills` on either platform.

Set location overrides before launch. There is no separate skills-path variable;
project-local `.loki/skills` is not searched. No registration command is needed.

For your own `review-changes` skill, create this layout:

```text
skills/
+-- review-changes/
    +-- SKILL.md
```

Use this UTF-8 content:

```markdown
---
description: Review the current diff for correctness and missing tests.
---
Review the current diff without changing files.
Prioritize correctness bugs, edge cases, and missing workflow coverage.
If an Args line is supplied, use it to narrow the review.
Report findings with file and line references; say when none are found.
```

The directory supplies the identifier, `review-changes`. Frontmatter is
optional: Loki reads a single-line `description:`, not arbitrary YAML metadata.
A `name:` field does not rename it.

One grouping level is supported: `skills/team/review-changes/SKILL.md` becomes
`team:review-changes`. If `skills/team/SKILL.md` exists, it registers `team`
instead of scanning children. Deeper trees are not recursively discovered;
ambiguous identifiers are omitted. Avoid names colliding with local commands,
whose handling takes precedence.

### Use and update a skill

Inside Loki or an editor's prompt, enter:

```text
/review-changes focus on error handling
```

For a grouped skill, use `/team:review-changes ...`. You can also ask the model
to use a named available skill. Loki advertises the catalog through its Skill
tool; the model loads a skill with that tool rather than the frontend
preloading it. Trailing arguments remain request text; they are not shell
arguments and do not trigger template substitution.

Skill slash requests are **model-mediated**, unlike local commands such as
`/status`. They need a working model connection, and use of the instructions
depends on the model. ACP clients can expose the advertised catalog in a command
menu; no separate skill RPC is required.

The load result identifies the skill's directory. References to supporting files
are relative to that directory, not the chat's cwd. They are not automatically
read; scripts are not automatically executed. Access still uses ordinary tools
and their permissions. A load reads at most **100,000 characters** and reports
truncation.

The model's catalog refreshes on the next turn and each Skill call rescans the
file. The terminal's visible startup list is a snapshot; ACP menus are announced
on opening/restoring a session rather than continuously watched. Reopen an ACP
session to refresh that menu. Loaded instructions remain conversation history;
there is no unload command. Start a new conversation for a fresh history.

## Status and account controls

### Observed response status

`/status` shows the last HTTP response headers observed for the current
connection. `/status all` includes all known connections; add `--json` for JSON.
These use saved/local observations, **not a live service query**.

Outside a session, inspect saved observations from Bash:

```bash
"$LOKI" status
"$LOKI" status --json
"$LOKI" status --endpoint https://example.com/v1/chat/completions
```

In PowerShell, invoke with `& $LOKI`. For ChatGPT subscriptions, observations
also show last-reported quotas/reset times when available. `window: 7 days`
means the allowance's duration, not time remaining. `[retained observation]`
marks older data. Account quota and conversation-context usage are different
quantities.

Observations save on normal exit; `/status save` saves sooner. Other sessions'
unsaved observations are not visible, and crashes can lose them. Changing to
another account can leave old status visible; use `/account` for a live query.

The saved file is `$XDG_STATE_HOME/loki/response-headers.json`, normally
`~/.local/state/loki/response-headers.json` on POSIX or
`%LOCALAPPDATA%\loki\response-headers.json` on Windows. Known secret headers are
redacted, but other sensitive data can remain: review before sharing. Invalid
files are reported and can be removed manually to start over. This is diagnostic
data, not a billing record.

### Live account actions

`/account` lists the active connection's controls. Selecting one makes a live
provider request:

- **ChatGPT subscription:** `usage` reports allowance windows/resets; `resets`
  lists banked reset credits and lets you redeem them.
- **OpenRouter:** `balance` reports the key's spend limit and prepaid credit.
- **DeepSeek:** `balance` reports prepaid balance.

Redeeming a reset is irreversible and never automatic. These results are not
saved into the response-header observations file. Read the available action and
confirmation before approving it.

## Troubleshooting and diagnostics

Start with the actual error message and the status area's model, cwd and mode.

- **No provider/model selected:** enter `/model`. Check that the intended key
  was available before startup or that subscription login succeeded. Old
  explicit endpoint/model settings can override the account-picker flow.
- **Asked for a change but nothing changed:** check the mode and whether the
  message ended in `?`. Questions, Explore and Plan restrict action tools.
- **Tool output seems missing:** it may be hidden by default; inspect the result
  explicitly or enable [tool-output display](#tool-output-display).
- **Waiting a long time:** inspect `/ps` and any retry notice. The enabled retry
  hook's overload wait starts at ten minutes by default; cancel to stop it.
- **Context limit reached:** see [compaction](#context-compaction). An endless
  retry policy does not make the context smaller or add quota.
- **Windows workspace refused:** configure/verify that exact directory with
  `loki-setup`. Check grants rather than assuming a nearby folder is configured.
- **Linux security initialization failed:** check permitted namespaces and
  `/proc` in your VM/container. Credential protection is not optional fallback.
- **Display or keyboard problems:** the terminal interface uses ANSI/ECMA-48
  screen controls. Check your terminal's support; Linux with foot is tested.
- **Keys or changed settings not visible in an editor:** restart its ACP agent
  with the intended startup environment; another terminal's exports do not
  update the running editor.

### Diagnostic logs

Developer traces -- unknown provider fields, timing, raw error payloads and
tracebacks -- are hidden by default. User-facing errors and hook stderr remain
visible. To enable debug logging to stderr in Bash:

```bash
LOKI_TRACE=1 "$LOKI"
# For the editor server, use the actual checkout's loki-acp launcher:
LOKI_TRACE=1 "$HOME/src/loki/loki-acp"
```

Alternatively set `LOKI_LOG_CONFIG` to an absolute path to a standard Python
logging INI file accessible to Loki and its helpers. It overrides `LOKI_TRACE`;
invalid/unreadable configuration prevents startup. Loki does not expand `~` in
that setting.

This POSIX example sends warnings to stderr and only provider-format debug
messages to a per-process file. Choose your own **private, writable** log path;
on Windows use a permitted location and Windows paths. Change
`loki_agent.formats` to `loki_agent` for all diagnostics or name another module,
such as `loki_agent.terminal_frontend`, for a narrower scope.

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
args=('/tmp/loki-trace-' + str(__import__('os').getpid()) + '.log', 'a', 'utf-8')

[formatter_diagnostic]
format=%(asctime)s %(levelname)s %(name)s[%(process)d]: %(message)s
```

Logging configuration can execute code: trust the file, not a configuration
supplied by a model or remote client. Relative log paths use the process's
working directory, not the INI file. Traces may contain conversation content and
provider secrets; review before sharing. Give each process its own file rather
than sharing a rotating log. Logger levels do not hide ordinary UI messages,
hook output or user-facing errors.

In agent-shell, server stderr appears in Notices even with its ACP logging
toggle off. A file destination avoids verbose notices. Never send ACP logs to
stdout, which carries editor communication. Helper-agent stderr can be inspected
through its job output.

## Developer reference

### ACP protocol notes

Loki supports negotiated ACP versions **1 and 2**. The editor and server must
agree on a version; changing the server does not automatically change the
client's requested version. The agent-shell checkout here initializes v1.

For compaction, v1 keeps the original prompt pending and separately replies to
each joined control when the shared operation settles; ordinary overlapping
prompts are rejected. Typed compaction progress needs the v1 client's
compaction capability.

With negotiated v2, prompt insertion is acknowledged independently of execution;
ordinary prompts can queue, and session-state/compaction updates report progress
and completion. Acknowledgement is not proof that inference or compaction has
finished. Cancelling queued execution does not erase an already acknowledged
user message. These are frontend protocol differences, not separate compaction
or model backends.

### Message and saved-log formats

ACP messages are limited to **16 MiB of UTF-8 JSON before the newline** on both
client/front and front/worker channels. Clients need this budget, not a library's
usual 64 KiB line limit. Loki splits text updates into complete frames without
dropping text. Structured results such as model-option lists are not silently
truncated; oversized results produce an RPC error.

A result-size error does not imply an operation was rolled back: do not blindly
replay a mutating request. Oversized input closes the channel after an error;
IDs that leave insufficient error-response space are rejected before dispatch
with a null-ID error. The frame limit does not bound total conversation memory.

Uncompacted saved logs use schema v4; logs with context replacement use v5, with
the original archive retained. Full decoding/validation happens on actual resume,
not when merely listing saved-session summaries. There is no automatic migration
of arbitrary older formats.

In ACP v1, Loki advertises skills through `session/update` with
`sessionUpdate: "available_commands_update"`. Selecting a skill sends its text
as an ordinary prompt. This v1 wire example is for client authors:

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "session/prompt",
  "params": {
    "sessionId": "YOUR_SESSION_ID",
    "prompt": [
      {"type": "text", "text": "/review-changes focus on error handling"}
    ]
  }
}
```

The model's Skill call uses an exact identifier without a slash and optional
arguments, for example:

```json
{"skill": "review-changes", "args": "focus on error handling"}
```

Neither path grants extra tools or permissions.

### Settings API and tests

Application code loads one `Settings` value through `loki_agent.settings`;
frontends do not parse settings files. Explicit layered loads let later options
override earlier ones. Hooks are identified by their section, and first
appearance fixes order. Terminal preference updates preserve unknown options and
hook definitions but not original comments/formatting.

Local development and CI use the same root-level parallel test driver. From the
Loki checkout:

```bash
python3 run_tests.py
python3 run_tests.py -p 'test_acp*' -j 4 -v -k prompt
```

Files run in separate Python processes. `-p` selects filenames, `-j` limits
workers, and `-s` selects a test directory. Unittest options such as `-v`, `-q`,
`-b`, `-f` and `-k` are forwarded. Live output is tagged by file, including
partial progress and stack diagnostics; a failing file makes the driver exit
nonzero.

Execution and diagnostic timers are **unlimited/off by default**.
`LOKI_SUITE_STALL_SECONDS=60` enables repeating worker stack dumps without
killing tests or changing results. `--timeout SECONDS` explicitly enables a
per-file execution limit. CI's job budget and diagnostic interval live in
`.github/workflows/python-app.yml`, not in test-driver defaults.
