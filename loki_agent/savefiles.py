"""On-disk chat-log format, path resolution, listing, and picker UI.

This module is the leaf data layer for session savefiles. It owns no
live agent state (transcript_items/session_todos/chat_log file handle
live in loki.py) and has no upward dependency on loki.py -- every
function that needs context takes it as a parameter.
"""

import json
import os
import re
import sys
import time
import uuid
from pprint import pformat

from . import formats


_PREVIEW_RE = re.compile(
    r'"role"\s*:\s*"user".*?"text"\s*:\s*"((?:\\.|[^"\\])*)',
    re.DOTALL,
)


def chat_log_filename(chat_id: str) -> str:
    if chat_id.startswith("chat-") and chat_id.endswith(".json"):
        return chat_id
    return "chat-{}.json".format(chat_id)


def ensure_chat_log_dir(chat_log_dir: str) -> None:
    os.makedirs(chat_log_dir, exist_ok=True)


def new_chat_log_path(chat_log_dir: str) -> str:
    ensure_chat_log_dir(chat_log_dir)
    return os.path.join(chat_log_dir, chat_log_filename(str(uuid.uuid4())))


def resolve_chat_log_path(resume_arg: str, startup_cwd: str,
                          chat_log_dir: str, resolve_path_fn) -> str:
    # Bare resume names are chat ids in the local Loki chat directory. An
    # absolute path or a path with a directory part is treated as a literal
    # path; a bare name is resolved inside the chat log directory.
    if os.path.isabs(resume_arg):
        return os.path.normpath(resume_arg)
    if os.path.dirname(resume_arg):
        return resolve_path_fn(resume_arg, startup_cwd)
    ensure_chat_log_dir(chat_log_dir)
    return os.path.join(chat_log_dir, chat_log_filename(resume_arg))


def chat_log_paths(chat_log_dir: str) -> list[str]:
    ensure_chat_log_dir(chat_log_dir)
    try:
        names = os.listdir(chat_log_dir)
    except FileNotFoundError:
        return []
    return [os.path.join(chat_log_dir, n) for n in names
            if n.startswith("chat-") and n.endswith(".json")]


def text_for(path: str) -> str:
    # Re-read each call; the kernel page cache keeps the bytes in RAM after
    # the first read, so repeated calls during a picker session are cheap.
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return ""
    return data.decode('utf-8', 'replace')


def preview(text: str) -> str:
    # Snatch a one-line preview from the first user message. The chat log's
    # first "text" field is always the system instruction ("You are a helpful
    # system agent..."), which is useless for distinguishing chats, so we anchor
    # on "role":"user" and grab the next "text" after it. Not JSON -- regex on
    # the decoded text, tolerant of anything weird mid-file.
    m = _PREVIEW_RE.search(text)
    snippet = m.group(1) if m else ""
    snippet = re.sub(r'\s+', ' ', snippet).strip()
    if len(snippet) > 80:
        snippet = snippet[:77] + "..."
    return snippet


def format_picker_row(idx: int, path: str) -> str:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    when = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
    chat_id = os.path.basename(path)[len("chat-"):-len(".json")]
    return f"  {idx}. {when}  {chat_id[:8]}  {preview(text_for(path))}"


def filtered_chat_log_paths(query: str, chat_log_dir: str) -> list[str]:
    # Match by whole-file substring, not parsed JSON -- fast, tolerant of
    # partially-written logs. mtime ordering puts the most recently touched
    # conversation at the bottom of the list.
    words = query.lower().split()
    paths = chat_log_paths(chat_log_dir)
    if not words:
        matches = list(paths)
    else:
        matches = []
        for path in paths:
            blob = text_for(path).lower()
            if all(w in blob for w in words):
                matches.append(path)
    matches.sort(key=os.path.getmtime)
    return matches


async def run_session_picker_async(*, input_fn, terminal, chat_log_dir: str):
    # Three gestures: "filter <words>" / bare "filter" sets/clears the filter;
    # a bare int selects that row; empty cancels (returns None -> caller starts
    # a fresh chat).
    query = ""
    while True:
        matches = filtered_chat_log_paths(query, chat_log_dir)
        print("Resume a chat. \"filter <words>\" searches the logs (words in any order);")
        print("bare \"filter\" clears it; a number selects that row; empty cancels.")
        if matches:
            for i, path in enumerate(matches, 1):
                print(format_picker_row(i, path))
        else:
            print("  (no chats match -- broaden your filter, or type \"filter\" to clear)")
        selection = await input_fn('number opens that row, "filter WORDS" narrows, empty cancels: ')
        s = (selection or "").strip()

        # Branch 1: "filter" command -- set the filter, or clear if bare.
        if s == 'filter' or s.startswith('filter '):
            query = s[len('filter'):].strip()
            continue

        # Branch 2: bare integer -- select item N from the current filtered view.
        try:
            n = int(s)
        except ValueError:
            n = None
        if n is not None:
            if 1 <= n <= len(matches):
                return matches[n - 1]
            continue  # out of range, re-render, keep current filter

        # Branch 3: empty -- cancel the picker.
        if not s:
            return None

        # Anything else -- re-render, keep current filter.
        continue


def read_chat_log(file_obj) -> tuple[list, list, dict]:
    blob = json.load(file_obj)
    transcript, todos = formats.load_log_blob(blob)
    state = blob.get("session_state", {}) if isinstance(blob, dict) else {}
    if not isinstance(state, dict):
        state = {}
    return transcript, todos, state


def write_chat_log(file_obj, transcript: list, todos: list,
                   session_state: dict) -> None:
    file_obj.seek(0)
    blob = formats.new_log_blob(transcript, todos)
    blob["session_state"] = session_state
    json.dump(blob, file_obj, indent=4)
    file_obj.truncate()
    file_obj.flush()
    print('Note: Saved chat log in {!r}'.format(file_obj.name), file=sys.stderr)
    sys.stderr.flush()


class ResumeTranscriptRenderer:
    """Render a loaded transcript as the previous terminal conversation."""

    def __init__(self, assistant_label: str = "Assistant"):
        self.assistant_label = assistant_label
        self.current_assistant_label = assistant_label

    def _message_label(self, item: dict) -> str:
        role = item.get("role")
        if role == "user":
            return "User"
        if role == "assistant":
            return self.current_assistant_label
        return str(role or "Message").capitalize()

    def _render_message(self, item: dict) -> str:
        text = formats.item_text(item).strip()
        if not text:
            return ""
        return f"{self._message_label(item)}: {text}"

    def _render_tool_call(self, item: dict) -> str:
        name = item.get("name") or "<unknown>"
        args = pformat(item.get("input", {}), width=100)
        return f"Tool call: {name}\n{args}"

    def _render_tool_result(self, item: dict) -> str:
        name = item.get("name") or item.get("tool_call_id") or "<unknown>"
        label = "Tool error" if item.get("is_error") else "Tool result"
        text = formats.item_text(item).strip()
        return f"{label}: {name}" + (f"\n{text}" if text else "")

    def _render_reasoning(self, item: dict) -> str:
        summary = item.get("summary")
        if not summary:
            return ""
        return "Reasoning summary:\n" + pformat(summary, width=100)

    def _render_provider_item(self, item: dict) -> str:
        provider = item.get("provider") or "unknown"
        return f"[Provider-specific transcript item: {provider}]\n{pformat(item.get('value'), width=100)}"

    def render_item(self, item: dict) -> str:
        item_type = item.get("type")
        if item_type == "response_metadata":
            if item.get("model"):
                self.current_assistant_label = item["model"]
            return ""
        if item_type == "instruction":
            return ""
        if item_type == "message":
            return self._render_message(item)
        if item_type == "tool_call":
            return self._render_tool_call(item)
        if item_type == "tool_result":
            return self._render_tool_result(item)
        if item_type == "reasoning":
            return self._render_reasoning(item)
        if item_type == "provider_item":
            return self._render_provider_item(item)
        return f"[Transcript item: {item_type or 'unknown'}]\n{pformat(item, width=100)}"

    def render(self, items: list) -> str:
        blocks = []
        for item in items:
            rendered = self.render_item(item)
            if rendered:
                blocks.append(rendered)
        return "\n\n".join(blocks)


def render_resume_transcript(items: list, assistant_label: str) -> str:
    return ResumeTranscriptRenderer(assistant_label=assistant_label).render(items)


def print_resume_transcript(items: list, assistant_label: str) -> None:
    rendered = render_resume_transcript(items, assistant_label)
    if rendered:
        print(rendered)
    print('----')
