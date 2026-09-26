"""Server usage observations and context display state, not token budgeting."""

from dataclasses import dataclass

from . import formats


def token_count(value):
    """Reject booleans, floats and malformed provider/catalog counts."""
    return value if type(value) is int and value >= 0 else None


@dataclass(frozen=True)
class ContextCapacity:
    tokens: int
    source: str

    def __post_init__(self):
        if token_count(self.tokens) is None or self.tokens == 0:
            raise ValueError("context capacity must be a positive integer")
        if self.source not in ("configured", "models.dev", "openai-subscription"):
            raise ValueError("unknown context capacity source")

    def to_dict(self):
        return {"tokens": self.tokens, "source": self.source}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("context capacity must be an object")
        return cls(value.get("tokens"), value.get("source"))


def catalog_capacity(value, source):
    count = token_count(value)
    return ContextCapacity(count, source) if count else None


@dataclass(frozen=True)
class ResponseUsage:
    input_tokens: int
    output_tokens: int
    # Future estimates can have their own source/basis without replacing the
    # server observation. No estimation or reconciliation is implemented here.
    source: str = "server"

    @property
    def used(self):
        return self.input_tokens + self.output_tokens


def normalize_usage(protocol, raw):
    """Normalize one exchange. Cache/reasoning breakdowns are not additive."""
    if not isinstance(raw, dict):
        return None
    if protocol == formats.OPENAI_CHAT:
        input_tokens = token_count(raw.get("prompt_tokens"))
        output_tokens = token_count(raw.get("completion_tokens"))
    elif protocol in (formats.OPENAI_RESPONSES, formats.ANTHROPIC_MESSAGES):
        input_tokens = token_count(raw.get("input_tokens"))
        output_tokens = token_count(raw.get("output_tokens"))
        if protocol == formats.ANTHROPIC_MESSAGES:
            # Anthropic input_tokens excludes cache hits and cache writes.
            # Nested cache_creation fields only break down the write total.
            cached = token_count(raw.get("cache_read_input_tokens", 0))
            created = token_count(raw.get("cache_creation_input_tokens", 0))
            if None in (input_tokens, cached, created):
                return None
            input_tokens += cached + created
    else:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    return ResponseUsage(input_tokens, output_tokens)


@dataclass(frozen=True)
class ContextSnapshot:
    usage: ResponseUsage | None
    capacity: ContextCapacity | None
    stale: bool = False

    @property
    def percentage(self):
        if self.usage is None or self.capacity is None:
            return None
        # Round half up using integer arithmetic; never hide values above 100%.
        return (self.usage.used * 200 + self.capacity.tokens) // (
            2 * self.capacity.tokens)

    @property
    def text(self):
        percentage = self.percentage
        if percentage is None:
            return "unknown"
        return f"{percentage}%" + ("*" if self.stale else "")


class ContextTracker:
    """Session-owned, derived snapshot of append-only canonical response events.

    No snapshot is persisted. Replacement (including resume) reconstructs it
    from raw events. Only newly appended events are inspected during a turn.
    Other transcripts, e.g. WebFetch helpers and child agents, never enter it.
    """

    def __init__(self):
        self._transcript = None
        self._identity = None
        self._processed = 0
        self._usage = None
        self._boundary = None
        self._historical = False

    def snapshot(self, transcript, identity, capacity, *, live=False):
        reset = (self._transcript is not transcript
                 or self._identity != identity
                 or len(transcript) < self._processed)
        if reset:
            self._transcript = transcript
            self._identity = identity
            self._processed = 0
            self._usage = None
            self._boundary = None
            self._historical = not live
        for index in range(self._processed, len(transcript)):
            event = transcript[index]
            if event.get("type") != "model_response":
                continue
            event_identity = (
                event.get("protocol"), event.get("provider"),
                event.get("endpoint"),
                event.get("requested_model", event.get("model")),
            )
            if identity is None or event_identity != identity:
                continue
            # Missing usage on a newer response must not look like a fresh
            # measurement from an older response.
            self._usage = normalize_usage(
                event.get("protocol"), event.get("usage"))
            self._boundary = index
            self._historical = not live or event.get("status") != "completed"
        self._processed = len(transcript)
        return ContextSnapshot(
            self._usage, capacity,
            self._historical or self._boundary != len(transcript) - 1,
        )
