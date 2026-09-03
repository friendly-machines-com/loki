"""On-disk chat-log format, path resolution, and presentation data.

This module owns no live agent state and no terminal policy. Picker output is
written through an injected text writer, and resume presentation is returned
as typed text segments for a front-end to render.
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


async def run_session_picker_async(
        *, input_fn, chat_log_dir: str, text_writer):
    # Three gestures: "filter <words>" / bare "filter" sets/clears the filter;
    # a bare int selects that row; empty cancels (returns None -> caller starts
    # a fresh chat).
    query = ""
    while True:
        matches = filtered_chat_log_paths(query, chat_log_dir)
        print()
        print("Saved sessions:")
        print("Resume a chat. \"filter <words>\" searches the logs (words in any order);")
        print("bare \"filter\" clears it; a number selects that row; empty cancels.")
        if matches:
            for i, path in enumerate(matches, 1):
                text_writer(format_picker_row(i, path))
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
                return matches[n - 1]
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
    sys.stdout.flush()
    print('Note: Saved chat log in {!r}'.format(path), file=sys.stderr)
    sys.stderr.flush()


def write_chat_log(file_obj, events: list, todos: list,
                   session_state: dict, toolsets=None) -> None:
    file_obj.seek(0)
    json.dump(
        chat_log_blob(
            events, todos, session_state, toolsets=toolsets),
        file_obj,
        indent=4,
    )
    file_obj.truncate()
    file_obj.flush()
    report_chat_log_saved(file_obj.name)


class ResumeTranscriptRenderer:
    """Describe and render the visible conversation in a loaded transcript.

    ``presentation`` keeps fixed punctuation, single-line dynamic atoms,
    multiline text, and assistant Markdown as separate segments. A terminal
    front-end can therefore apply its own output policy without this shared
    module importing terminal code.
    """

    def __init__(self, assistant_label: str = "Assistant",
                 assistant_text_renderer=None, user_text_renderer=None):
        self.assistant_label = assistant_label
        self.assistant_text_renderer = (
            assistant_text_renderer
            if assistant_text_renderer is not None
            else (lambda text: text))
        self.user_text_renderer = (
            user_text_renderer
            if user_text_renderer is not None
            else (lambda text: text))

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
            blocks.append([
                label_segment,
                ("literal", ": "),
                (content_kind, text),
            ])
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
                blocks.append([("literal", marker)])
        return blocks

    @staticmethod
    def _tool_call_blocks(item: dict):
        name = formats.tool_call_name(item) or "<unknown>"
        if item.get("execution", "client") != "client":
            return [[
                ("literal", "Provider tool call: "),
                ("program_atom", name),
                ("literal", "\n"),
                ("text", pformat(item, width=100)),
            ]]
        try:
            input_value = formats.tool_call_input(item)
        except formats.TranscriptFormatError:
            input_value = item.get("arguments", item.get("input", {}))
        return [[
            ("literal", "Tool call: "),
            ("program_atom", name),
            ("literal", "\n"),
            ("text", pformat(input_value, width=100)),
        ]]

    @staticmethod
    def _tool_result_blocks(item: dict):
        name = item.get("name") or item.get("call_id") or "<unknown>"
        label = "Tool error" if item.get("is_error") else "Tool result"
        text = formats.item_text(item).strip()
        block = [
            ("literal", label + ": "),
            ("program_atom", name),
        ]
        if text:
            block.extend([("literal", "\n"), ("text", text)])
        return [block]

    @staticmethod
    def _reasoning_blocks(item: dict):
        value = item.get("value", item)
        summary = value.get("summary") if isinstance(value, dict) else None
        if not summary:
            return []
        return [[
            ("literal", "Reasoning summary:\n"),
            ("text", pformat(summary, width=100)),
        ]]

    @staticmethod
    def _native_item_blocks(item: dict, response_protocol=None):
        provider = (
            item.get("protocol")
            or item.get("provider")
            or response_protocol
            or "unknown")
        return [[
            ("literal", "[Provider-specific transcript item: "),
            ("program_atom", provider),
            ("literal", "]\n"),
            ("text", pformat(item.get("value"), width=100)),
        ]]

    @staticmethod
    def _provider_tool_result_blocks(item: dict, response_protocol=None):
        provider = response_protocol or "provider"
        call_id = item.get("call_id") or "<unknown>"
        return [[
            ("literal", "Provider tool result ("),
            ("program_atom", provider),
            ("literal", "): "),
            ("program_atom", call_id),
            ("literal", "\n"),
            ("text", pformat(item.get("content"), width=100)),
        ]]

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
        return [block]

    def _response_blocks(self, event: dict):
        label = event.get("model") or self.assistant_label
        blocks = []
        for code in formats.provider_notice_codes(event):
            text = formats.provider_notice_text(code)
            if text is not None:
                blocks.append([
                    ("literal", "\u24d8 "),
                    ("text", text),
                ])
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
            blocks.append(block)
        for item in event.get("items", []):
            item_type = item.get("type")
            if item_type == "message":
                rendered = self._message_blocks(item, label=label)
            elif item_type == "function_call":
                rendered = self._tool_call_blocks(item)
            elif item_type == "openai_reasoning":
                rendered = self._reasoning_blocks(item)
            elif item_type in [
                    "anthropic_thinking",
                    "anthropic_redacted_thinking"]:
                rendered = []
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
                rendered = [[
                    ("literal", "[Model response item: "),
                    ("atom", str(item_type or "unknown")),
                    ("literal", "]\n"),
                    ("text", pformat(item, width=100)),
                ]]
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
        return [[
            ("literal", "[Session event: "),
            ("atom", str(item_type or "unknown")),
            ("literal", "]\n"),
            ("text", pformat(item, width=100)),
        ]]

    def presentation(self, events: list):
        blocks = []
        for event in events:
            blocks.extend(self._event_blocks(event))
        return blocks

    def render(self, events: list) -> str:
        rendered_blocks = []
        for block in self.presentation(events):
            rendered_segments = []
            for kind, text in block:
                if kind == "assistant_markdown":
                    text = self.assistant_text_renderer(text)
                elif kind == "user_text":
                    text = self.user_text_renderer(text)
                rendered_segments.append(text)
            rendered_blocks.append("".join(rendered_segments))
        return "\n\n".join(rendered_blocks)


def render_resume_transcript(
        events: list, assistant_label: str,
        assistant_text_renderer=None, user_text_renderer=None) -> str:
    return ResumeTranscriptRenderer(
        assistant_label=assistant_label,
        assistant_text_renderer=assistant_text_renderer,
        user_text_renderer=user_text_renderer,
    ).render(events)
