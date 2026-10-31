"""On-disk chat-log format, path resolution, and presentation data.

This module owns no live agent state and no terminal policy. Picker output is
written through an injected text writer, and resume presentation is returned
as typed text segments for a front-end to render.
"""

from contextlib import contextmanager
import json
import os
import re
import stat
# import sys
import time
import uuid
from pprint import pformat

from . import formats
from . import process_outputs
from . import texts


_PREVIEW_RE = re.compile(
    r'"role"\s*:\s*"user".*?"text"\s*:\s*"((?:\\.|[^"\\])*)',
    re.DOTALL,
)


def chat_log_filename(chat_id: str) -> str:
    if chat_id.startswith("chat-") and chat_id.endswith(".json"):
        return chat_id
    return "chat-{}.json".format(chat_id)


def new_chat_log_path(chat_log_dir: str) -> str:
    return os.path.join(chat_log_dir, chat_log_filename(str(uuid.uuid4())))


def resolve_chat_log_path(resume_arg: str, startup_cwd: str,
                          chat_log_dir: str, resolve_path_fn) -> str:
    # This CLI argument admits two namespaces: bare saved names and explicit
    # filenames. Only this grammar decision needs the native directory/drive
    # view. The original operand, including every dot component, is forwarded.
    directory, _name = os.path.split(resume_arg)
    if directory:
        return resolve_path_fn(resume_arg, startup_cwd)
    return os.path.join(chat_log_dir, chat_log_filename(resume_arg))


def chat_log_entries(chat_log_dir: str) -> list[os.DirEntry]:
    try:
        with os.scandir(chat_log_dir) as directory:
            return [entry for entry in directory
                    if entry.name.startswith("chat-") and entry.name.endswith(".json")]
    except FileNotFoundError:
        return []


@contextmanager
def open_chat_log(path: str, *, errors='strict'):
    """Validate the opened saved conversation, without a pathname recheck.

    A regular-file symlink remains usable. POSIX special files must not block
    a frontend before its regular-file check can run.
    """
    flags = os.O_RDONLY
    if os.name == 'posix':
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        facts = os.fstat(descriptor)
        if not stat.S_ISREG(facts.st_mode):
            raise OSError('saved conversation is not a regular file')
        stream = os.fdopen(descriptor, 'r', encoding='utf-8', errors=errors)
        descriptor = None
        with stream:
            yield stream, facts
    finally:
        if descriptor is not None:
            os.close(descriptor)


def text_for(path: str) -> str:
    # Re-read each call rather than retaining a second copy of a saved log.
    try:
        with open_chat_log(path, errors='replace') as (stream, _facts):
            return stream.read()
    except OSError:
        return ""


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


def format_picker_row(idx: int, entry: os.DirEntry) -> str:
    try:
        mtime = os.path.getmtime(entry.path)
    except OSError:
        mtime = 0
    when = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
    chat_id = entry.name[len("chat-"):-len(".json")]
    return f"  {idx}. {when}  {chat_id[:8]}  {preview(text_for(entry.path))}"


def filtered_chat_log_entries(query: str, chat_log_dir: str) -> list[os.DirEntry]:
    # Match by whole-file substring, not parsed JSON -- fast, tolerant of
    # partially-written logs. mtime ordering puts the most recently touched
    # conversation at the bottom of the list.
    words = query.lower().split()
    entries = chat_log_entries(chat_log_dir)
    if not words:
        matches = entries
    else:
        matches = []
        for entry in entries:
            blob = text_for(entry.path).lower()
            if all(w in blob for w in words):
                matches.append(entry)
    matches.sort(key=lambda entry: os.path.getmtime(entry.path))
    return matches


async def run_session_picker_async(
        *, input_fn, chat_log_dir: str, text_writer):
    # Three gestures: "filter <words>" / bare "filter" sets/clears the filter;
    # a bare int selects that row; empty cancels (returns None -> caller starts
    # a fresh chat).
    query = ""
    while True:
        matches = filtered_chat_log_entries(query, chat_log_dir)
        print()
        print("Saved sessions:")
        print("Resume a chat. \"filter <words>\" searches the logs (words in any order);")
        print("bare \"filter\" clears it; a number selects that row; empty cancels.")
        if matches:
            for i, entry in enumerate(matches, 1):
                text_writer(format_picker_row(i, entry))
                print()
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
                return matches[n - 1].path
            continue  # out of range, re-render, keep current filter

        # Branch 3: empty -- cancel the picker.
        if not s:
            return None

        # Anything else -- re-render, keep current filter.
        continue


def read_chat_log(file_obj) -> tuple[list, list, dict, list]:
    blob = json.load(file_obj)
    events, todos = formats.load_log_blob(blob)
    state = blob.get("session_state", {}) if isinstance(blob, dict) else {}
    if not isinstance(state, dict):
        state = {}
    return (
        events,
        todos,
        state,
        formats.log_toolsets(blob),
    )


def chat_log_blob(events: list, todos: list,
                  session_state: dict, toolsets=None) -> dict:
    blob = formats.new_log_blob(
        events, todos, toolsets=toolsets)
    blob["session_state"] = session_state
    return blob


def serialize_chat_log(events: list, todos: list,
                       session_state: dict, toolsets=None) -> str:
    return json.dumps(
        chat_log_blob(
            events, todos, session_state, toolsets=toolsets),
        indent=4,
    )


def report_chat_log_saved(path: str) -> None:
    # sys.stdout.flush()
    # print('Note: Saved chat log in {!r}'.format(path), file=sys.stderr)
    # sys.stderr.flush()
    pass


class ResumeTranscriptRenderer:
    """Describe and render the visible conversation in a loaded transcript.

    ``presentation`` keeps fixed punctuation, single-line dynamic atoms,
    multiline text, and assistant Markdown as separate segments. A terminal
    front-end can therefore apply its own output policy without this shared
    module importing terminal code.
    """

    def __init__(self, assistant_label: str = "Assistant", *, show_reasoning=False,
                 show_bash_stdout=None, show_read_stdout=None, show_skill_stdout=None):
        self.assistant_label = assistant_label
        self.show_reasoning = show_reasoning
        # None preserves the canonical text used by non-terminal consumers.
        self.show_bash_stdout = show_bash_stdout
        self.show_read_stdout = show_read_stdout
        self.show_skill_stdout = show_skill_stdout

    def _message_label(self, item: dict):
        role = item.get("role")
        if role == "user":
            return "literal", "User"
        if role == "assistant":
            return "atom", self.assistant_label
        return "atom", str(role or "Message").capitalize()

    def _message_blocks(self, item: dict, label=None):
        blocks = []
        text = formats.item_text(item)
        if text.strip():
            label_segment = (
                ("atom", label) if label is not None
                else self._message_label(item))
            role = item.get("role")
            content_kind = (
                "assistant_markdown" if role == "assistant"
                else "user_text" if role == "user"
                else "text")
            blocks.append(("message", [
                label_segment,
                ("literal", ": "),
                (content_kind, text),
            ]))
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            marker = {
                "image": "[Image content]",
                "file": "[File content]",
                "document": "[File content]",
                "audio": "[Audio content]",
            }.get(content.get("type"))
            if marker:
                blocks.append(("message", [("literal", marker)]))
        return blocks

    @staticmethod
    def _tool_call_blocks(item: dict):
        name = formats.tool_call_name(item) or "<unknown>"
        if item.get("execution", "client") != "client":
            return [("provider", [
                ("literal", "Provider tool call: "),
                ("program_atom", name),
                ("literal", "\n"),
                ("text", pformat(item, width=100)),
            ])]
        try:
            input_value = formats.tool_call_input(item)
        except formats.TranscriptFormatError:
            input_value = item.get("arguments", item.get("input", {}))
        return [("tool_call", [
            ("literal", "Tool call: "),
            ("program_atom", name),
            ("literal", "\n"),
            ("text", texts.format_tool_args(input_value)),
        ])]

    def _tool_result_blocks(self, item: dict):
        name = item.get("name") or item.get("call_id") or "<unknown>"
        label = "Tool error" if item.get("is_error") else "Tool result"
        text = formats.item_text(item).strip()
        if (self.show_bash_stdout is not None or self.show_read_stdout is not None
                or self.show_skill_stdout is not None):
            text = process_outputs.presentation_text(
                text, item.get("process_output"),
                show_bash_stdout=True if self.show_bash_stdout is None else self.show_bash_stdout,
                show_read_stdout=True if self.show_read_stdout is None else self.show_read_stdout,
                show_skill_stdout=True if self.show_skill_stdout is None else self.show_skill_stdout,
                legacy_bash=name == "Bash" and self.show_bash_stdout is not None,
                legacy_read=name == "Read" and self.show_read_stdout is not None
                and not item.get("is_error"))
        block = [
            ("literal", label + ": "),
            ("program_atom", name),
        ]
        if text:
            block.extend([("literal", "\n"), ("text", text)])
        return [("tool_error" if item.get("is_error") else "tool_result",
                 block)]

    def _reasoning_blocks(self, item: dict, response: dict):
        if not self.show_reasoning:
            return []
        return [("reasoning", [
            ("literal", "Reasoning summary:\n" if kind == "summary" else "Thinking:\n"),
            ("text", text),
        ]) for key, kind, text in formats.reasoning_segments(
            [item], source=formats.reasoning_field(response))]

    @staticmethod
    def _native_item_blocks(item: dict, response_protocol=None):
        provider = (
            item.get("protocol")
            or item.get("provider")
            or response_protocol
            or "unknown")
        return [("provider_item", [
            ("literal", "[Provider-specific transcript item: "),
            ("program_atom", provider),
            ("literal", "]\n"),
            ("text", pformat(item.get("value"), width=100)),
        ])]

    @staticmethod
    def _provider_tool_result_blocks(item: dict, response_protocol=None):
        provider = response_protocol or "provider"
        call_id = item.get("call_id") or "<unknown>"
        return [("provider", [
            ("literal", "Provider tool result ("),
            ("program_atom", provider),
            ("literal", "): "),
            ("program_atom", call_id),
            ("literal", "\n"),
            ("text", pformat(item.get("content"), width=100)),
        ])]

    @staticmethod
    def _provider_operation_blocks(item: dict, response_protocol=None):
        provider = response_protocol or "provider"
        name = item.get("name") or "<unknown>"
        block = [
            ("literal", "Provider operation ("),
            ("program_atom", provider),
            ("literal", "): "),
            ("program_atom", name),
            ("literal", "\n"),
            ("text", pformat(item.get("input"), width=100)),
        ]
        if item.get("output") is not None:
            block.extend([
                ("literal", "\nProvider operation result:\n"),
                ("text", pformat(item.get("output"), width=100)),
            ])
        return [("provider", block)]

    def _response_blocks(self, event: dict):
        label = event.get("model") or self.assistant_label
        blocks = []
        for code in formats.provider_notice_codes(event):
            text = formats.provider_notice_text(code)
            if text is not None:
                blocks.append(("notice", [
                    ("literal", "\u24d8 "),
                    ("text", text),
                ]))
        status = event.get("status", "completed")
        if status != "completed":
            block = [
                ("literal", "[Model response "),
                ("atom", str(status)),
                ("literal", "]"),
            ]
            detail = event.get("protocol_data")
            if detail:
                block.extend([
                    ("literal", "\n"),
                    ("text", pformat(detail, width=100)),
                ])
            blocks.append(("status", block))
        for item in event.get("items", []):
            item_type = item.get("type")
            if item_type == "message":
                rendered = self._reasoning_blocks(item, event) + self._message_blocks(item, label=label)
            elif item_type == "function_call":
                rendered = self._tool_call_blocks(item)
            elif item_type == "openai_reasoning":
                rendered = self._reasoning_blocks(item, event)
            elif item_type in [
                    "anthropic_thinking",
                    "anthropic_redacted_thinking"]:
                rendered = self._reasoning_blocks(item, event)
            elif item_type in ["native_output", "provider_output"]:
                rendered = self._native_item_blocks(
                    item, response_protocol=event.get("protocol"))
            elif item_type == "provider_tool_result":
                rendered = self._provider_tool_result_blocks(
                    item, response_protocol=event.get("protocol"))
            elif item_type == "provider_operation":
                rendered = self._provider_operation_blocks(
                    item, response_protocol=event.get("protocol"))
            else:
                rendered = [("other", [
                    ("literal", "[Model response item: "),
                    ("atom", str(item_type or "unknown")),
                    ("literal", "]\n"),
                    ("text", pformat(item, width=100)),
                ])]
            blocks.extend(rendered)
        return blocks

    def _event_blocks(self, item: dict):
        item_type = item.get("type")
        if item_type == "message" and item.get("role") in [
                "system", "developer"]:
            return []
        if item_type == "message":
            return self._message_blocks(item)
        if item_type == "model_response":
            return self._response_blocks(item)
        if item_type == "tool_result":
            return self._tool_result_blocks(item)
        return [("other", [
            ("literal", "[Session event: "),
            ("atom", str(item_type or "unknown")),
            ("literal", "]\n"),
            ("text", pformat(item, width=100)),
        ])]

    def presentation(self, events: list):
        blocks = []
        for event in events:
            blocks.extend(self._event_blocks(event))
        return blocks
