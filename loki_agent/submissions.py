"""Frontend input identity, independent of any transport or terminal."""

from dataclasses import dataclass, field
import uuid


@dataclass(frozen=True)
class Submission:
    text: str
    origin: str = "keyboard"
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def normalize(cls, value):
        return value if isinstance(value, cls) else cls(value)
