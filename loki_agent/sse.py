"""Small incremental Server-Sent Events decoder."""

import codecs
from dataclasses import dataclass


SSE_MAX_EVENT_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: str
    event_id: str | None = None


class SseDecoder:
    """Decode arbitrarily fragmented UTF-8 SSE bytes into complete events."""

    def __init__(self, max_event_bytes=SSE_MAX_EVENT_BYTES):
        self.max_event_bytes = max_event_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._buffer = ""
        self._data_lines = []
        self._event_type = ""
        self._event_id = None
        self._event_bytes = 0
        self._first_line = True

    def feed(self, chunk: bytes) -> list[SseEvent]:
        if not isinstance(chunk, bytes):
            raise TypeError("SSE input chunks must be bytes")
        self._buffer += self._decoder.decode(chunk, final=False)
        self._check_buffer_size()
        return self._consume_lines(final=False)

    def finish(self) -> list[SseEvent]:
        self._buffer += self._decoder.decode(b"", final=True)
        events = self._consume_lines(final=True)
        event = self._dispatch()
        if event is not None:
            events.append(event)
        return events

    def _check_buffer_size(self):
        if len(self._buffer.encode("utf-8")) > self.max_event_bytes:
            raise ValueError("SSE line exceeds event size limit")

    def _consume_lines(self, final: bool) -> list[SseEvent]:
        events = []
        while self._buffer:
            boundary = None
            separator_length = 0
            for index, char in enumerate(self._buffer):
                if char == "\n":
                    boundary = index
                    separator_length = 1
                    break
                if char == "\r":
                    if index + 1 == len(self._buffer) and not final:
                        return events
                    boundary = index
                    separator_length = (
                        2 if self._buffer[index:index + 2] == "\r\n" else 1)
                    break
            if boundary is None:
                if not final:
                    return events
                line = self._buffer
                self._buffer = ""
            else:
                line = self._buffer[:boundary]
                self._buffer = self._buffer[boundary + separator_length:]
            event = self._consume_line(line)
            if event is not None:
                events.append(event)
        return events

    def _consume_line(self, line: str) -> SseEvent | None:
        if self._first_line:
            line = line.removeprefix("\ufeff")
            self._first_line = False
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        self._event_bytes += len(line.encode("utf-8"))
        if self._event_bytes > self.max_event_bytes:
            raise ValueError("SSE event exceeds size limit")
        if field == "data":
            self._data_lines.append(value)
        elif field == "event":
            self._event_type = value
        elif field == "id" and "\0" not in value:
            self._event_id = value
        return None

    def _dispatch(self) -> SseEvent | None:
        if not self._data_lines:
            self._event_type = ""
            self._event_bytes = 0
            return None
        event = SseEvent(
            self._event_type or "message",
            "\n".join(self._data_lines),
            self._event_id,
        )
        self._data_lines = []
        self._event_type = ""
        self._event_bytes = 0
        return event


async def iter_sse_events(byte_chunks, max_event_bytes=SSE_MAX_EVENT_BYTES):
    decoder = SseDecoder(max_event_bytes=max_event_bytes)
    async for chunk in byte_chunks:
        for event in decoder.feed(chunk):
            yield event
    for event in decoder.finish():
        yield event
