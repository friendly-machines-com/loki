"""Per-conversation state.

A Session holds everything one conversation owns: its virtual cwd, the
active provider connection, the transcript, todos, and chat-log
bookkeeping.  Front-ends pass it along explicitly; terminal and headless
modes use exactly one for the life of the process.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any


def prompt_cache_key_for_path(path: str) -> str:
    """Return a stable, non-secret cache partition for one persistent chat."""
    real_path = os.path.realpath(path)
    name = os.path.basename(real_path)
    if name.startswith("chat-") and name.endswith(".json"):
        chat_id = name[len("chat-"):-len(".json")]
        # Terminal chats use a bare UUID; ACP-created chats use "loki-UUID".
        candidate = (
            chat_id[len("loki-"):]
            if chat_id.startswith("loki-") else chat_id)
        try:
            return str(uuid.UUID(candidate))
        except ValueError:
            pass
    # Explicitly named chat files still need the same partition after resume.
    # uuid5 avoids sending the absolute path itself to the model service.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, real_path))


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
    # One cache partition belongs to one conversation. It is derived again
    # from persistent chat identity on resume rather than trusting an
    # arbitrary value in the chat log. A nonpersistent runtime keeps this
    # fresh in-memory value for its own lifetime.
    #
    # This is not x-codex-turn-state: that routing token is created afresh for
    # every logical user turn.
    prompt_cache_key: str = field(
        default_factory=lambda: str(uuid.uuid4()))

    # "normal" / "explore" / "plan" / "edit"
    agent_mode: str = "normal"
    last_instructed_agent_mode: str | None = None

    # Delegation depth is runtime ownership state, not transcript state. Root
    # terminal and ACP sessions start at zero; an internal subagent entrypoint
    # installs the explicit depth delegated by its parent.
    subagent_depth: int = 0

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
        self.prompt_cache_key = prompt_cache_key_for_path(path)
        # Write through the real path, not a symlink naming it.
        self.chat_log_path = os.path.realpath(path) if path else None
        self.chat_log_dirty = False
        self.last_instructed_agent_mode = None


def default_session() -> Session:
    return Session()
