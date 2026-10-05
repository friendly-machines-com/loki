"""Captured stream identity for tool-result presentation.

The ordinary tool content remains the existing bounded text consumed by models,
hooks and ACP. This snapshot keeps the captured channels available to terminal
renderers and saved-transcript replay, even when that text projection truncates
before stderr. Nothing here executes commands or trusts saved presentation data
as process state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ProcessOutput:
    header: str
    stdout: str
    stderr: str
    shell: bool = True
    tail: bool = False
    preamble: str = ""
    read: bool = False

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
        for name in ["shell", "tail", "read"]:
            if name in value and type(value[name]) is not bool:
                raise ValueError(f"process output {name} must be boolean")
        return cls(**value)

    def render(self, *, show_stdout=True):
        parts = [self.header] if self.header else []
        suffix = "_tail" if self.tail else ""
        if show_stdout and self.stdout:
            parts.extend([self.stdout] if self.read else [f"[stdout{suffix}]", self.stdout])
        if self.stderr:
            parts.extend([f"[stderr{suffix}]", self.stderr])
        return self.preamble + "\n".join(parts)


def presentation_text(content, process_output=None, *, show_bash_stdout=False,
                      legacy_bash=False, show_read_stdout=False, legacy_read=False):
    """Select a view without mutating canonical content or guessing boundaries."""
    if process_output is not None:
        output = ProcessOutput.from_dict(process_output)
        if output.read:
            return output.render(show_stdout=show_read_stdout)
        if output.shell:
            return output.render(show_stdout=show_bash_stdout)
    if legacy_bash and not show_bash_stdout:
        return "Older combined Bash output is hidden; enable show_bash_stdout to display it."
    if legacy_read and not show_read_stdout:
        return "Older combined Read output is hidden; enable show_read_stdout to display it."
    return str(content)
