"""Per-conversation state.

A Session holds everything one conversation owns: its virtual cwd, the
active provider connection, the transcript, todos, and chat-log
bookkeeping.  Front-ends pass it along explicitly; terminal and headless
modes use exactly one for the life of the process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    # "Virtual" cwd: base for resolving relative tool input.  The process
    # cwd never changes; subprocesses get an explicit cwd= argument.
    shell_cwd: str = ""
    previous_shell_cwd: str = ""

    # RuntimeConfig, or None before startup config is applied.
    runtime_config: Any = None

    # Async request-time credential authority. Top-level processes install a
    # local broker; workers and subagents install a delegated client.
    credential_authority: Any = None

    transcript_items: list = field(default_factory=list)
    session_todos: list = field(default_factory=list)
    session_toolsets: list = field(default_factory=list)

    session_state: dict = field(default_factory=dict)
    chat_log_path: str | None = None
    chat_log_dirty: bool = False

    # "normal" / "explore" / "plan" / "edit"
    agent_mode: str = "normal"
    last_instructed_agent_mode: str | None = None

    # JobManager, created on first use.
    job_manager: Any = None

    def __post_init__(self):
        if not self.shell_cwd:
            self.shell_cwd = os.getcwd()
        if not self.previous_shell_cwd:
            self.previous_shell_cwd = self.shell_cwd

    @property
    def model(self) -> str:
        if self.runtime_config is None:
            return ""
        return self.runtime_config.model

    def replace_transcript(self, transcript, todos, toolsets, state, path):
        self.transcript_items = transcript
        self.session_todos = todos
        self.session_toolsets = toolsets
        self.session_state = dict(state)
        # Write through the real path, not a symlink naming it.
        self.chat_log_path = os.path.realpath(path) if path else None
        self.chat_log_dirty = False
        self.last_instructed_agent_mode = None


def default_session() -> Session:
    return Session()
