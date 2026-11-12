"""Captured stream identity for tool-result presentation.

Tool content is text or canonical text/media blocks consumed by models and
hooks. ProcessOutput keeps captured text channels available to renderers and
saved-transcript replay. When no captured channels exist, presentation_text
supplies text and media placeholders, not a repr of inline image data. The TUI,
ACP notifications and replay borrow this projection; the canonical model/history
content remains intact. Nothing here trusts presentation data as process state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


MEDIA_PLACEHOLDERS = {
    "image": "[Image content]",
    "file": "[File content]",
    "document": "[File content]",
    "audio": "[Audio content]",
}


@dataclass(frozen=True)
class ProcessOutput:
    header: str
    stdout: str
    stderr: str
    shell: bool = True
    tail: bool = False
    preamble: str = ""
    read: bool = False
    skill: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("process output must be an object")
        allowed = {entry.name for entry in fields(cls)}
        if set(value) - allowed or not {"header", "stdout", "stderr"} <= set(value):
            raise ValueError("invalid process output fields")
        for name in ["header", "stdout", "stderr", "preamble"]:
            if name in value and not isinstance(value[name], str):
                raise ValueError(f"process output {name} must be text")
        for name in ["shell", "tail", "read", "skill"]:
            if name in value and type(value[name]) is not bool:
                raise ValueError(f"process output {name} must be boolean")
        return cls(**value)

    def render(self, *, show_stdout=True):
        parts = [self.header] if self.header else []
        suffix = "_tail" if self.tail else ""
        if show_stdout and self.stdout:
            parts.extend([self.stdout] if self.read or self.skill else [f"[stdout{suffix}]", self.stdout])
        if self.stderr:
            parts.extend([f"[stderr{suffix}]", self.stderr])
        return self.preamble + "\n".join(parts)


def presentation_text(content, process_output=None, *, show_bash_stdout=False,
                      legacy_bash=False, show_read_stdout=False, legacy_read=False,
                      show_skill_stdout=False):
    """Select a view without mutating canonical content or guessing boundaries."""
    if process_output is not None:
        output = ProcessOutput.from_dict(process_output)
        if output.read:
            return output.render(show_stdout=show_read_stdout)
        if output.skill:
            return output.render(show_stdout=show_skill_stdout)
        if output.shell:
            return output.render(show_stdout=show_bash_stdout)
    media = []
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            marker = MEDIA_PLACEHOLDERS.get(block.get("type"))
            if marker is not None:
                media.append(marker)
                parts.append(marker)
            elif block.get("type") in ["text", "input_text", "output_text", "refusal"]:
                text = block.get("text", block.get("refusal", ""))
                if isinstance(text, str) and text:
                    parts.append(text)
        text = "\n".join(parts)
    else:
        text = str(content)
    if legacy_bash and not show_bash_stdout:
        return "\n".join(media) or "Older combined Bash output is hidden; enable show_bash_stdout to display it."
    if legacy_read and not show_read_stdout:
        return "\n".join(media) or "Older combined Read output is hidden; enable show_read_stdout to display it."
    return text


def truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... [output truncated: {len(s)} chars total, {max_chars} shown]"
