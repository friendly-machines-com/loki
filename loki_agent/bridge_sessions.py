"""Session routing, bounded replay, and command identity for local bridges."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import uuid

from .bridge_transports import BridgeTransport, Limits, encode
from .submissions import Submission


@dataclass
class InputRecord:
    digest: str
    status: str = "queued"
    outcome: str | None = None


class BridgeSession:
    """One live runtime, not one saved conversation and not one socket lifetime.

    The frontend owns execution and cancellation. The proxy owns remote-user
    authorization and archive deduplication. Nothing here trusts an XMPP name
    or grants access to credentials, modal input, or another session.
    """

    def __init__(self, conversation_id, *, frontend, path, enqueue=None,
                 peer_uid=None, limits=None, report=None):
        self.conversation_id = conversation_id
        self.frontend = frontend
        self.instance_id = str(uuid.uuid4())
        self.enqueue = enqueue
        self.limits = limits or Limits()
        self.inputs = OrderedDict()
        self.sequence = 0
        self.acknowledged = 0
        self.journal = deque()
        self._journal_bytes = 0
        self.active = None
        self.paused = False
        self.closing = False
        self.transport = BridgeTransport(
            path, register=self.registration, receive=self.receive,
            limits=self.limits, peer_uid=peer_uid, report=report)

    async def __aenter__(self):
        await self.transport.__aenter__()
        return self

    async def __aexit__(self, *exc):
        self.closing = True
        for input_id, record in list(self.inputs.items()):
            if record.status == "queued":
                self.command_finished(
                    Submission("", "bridge", input_id), "unexecuted",
                    "Session closed before execution.")
        self.publish("session_closing")
        try:
            await self.transport.flush()
        finally:
            await self.transport.close()

    def registration(self):
        earliest = (self.journal[0][0]["event_seq"] if self.journal
                    else self.sequence + 1)
        messages = [{
            "type": "session_registered",
            "conversation_id": self.conversation_id,
            "instance_id": self.instance_id,
            "frontend": self.frontend,
            "label": self.conversation_id[:8],
            "capabilities": {"accepts_prompts": self.enqueue is not None,
                             "publishes_turns": True},
            "event_seq": self.sequence,
            "replay_from": earliest,
            "acknowledged": self.acknowledged,
            "paused": self.paused,
            "active": self.active,
            "inputs": [{"input_id": key, "status": record.status,
                        "outcome": record.outcome}
                       for key, record in self.inputs.items()],
        }]
        if earliest > self.acknowledged + 1:
            messages.append({"type": "event_gap",
                             "instance_id": self.instance_id,
                             "from_seq": self.acknowledged + 1,
                             "to_seq": earliest - 1})
        messages.extend(message.copy() for message, _size in self.journal)
        return messages

    def publish(self, event_type, **fields):
        self.sequence += 1
        message = {"type": event_type, "instance_id": self.instance_id,
                   "event_seq": self.sequence, **fields}
        size = len(encode(message, self.limits))
        self.journal.append((message, size))
        self._journal_bytes += size
        while (len(self.journal) > self.limits.events
               or self._journal_bytes > self.limits.buffer_bytes):
            _message, removed = self.journal.popleft()
            self._journal_bytes -= removed
        self.transport.send(message)

    def _reject(self, input_id, reason):
        self.publish("prompt_rejected", input_id=input_id, reason=reason)

    def receive(self, message):
        input_id = message.get("input_id")
        if message.get("instance_id") != self.instance_id:
            self._reject(input_id if isinstance(input_id, str)
                         and len(input_id) <= 128 else None,
                         "wrong_instance")
            return
        if message["type"] == "ack":
            sequence = message.get("event_seq")
            if (type(sequence) is not int
                    or not self.acknowledged <= sequence <= self.sequence):
                self.publish("protocol_error", reason="invalid_ack")
                return
            self.acknowledged = sequence
            while (self.journal
                   and self.journal[0][0]["event_seq"] <= sequence):
                _message, size = self.journal.popleft()
                self._journal_bytes -= size
            return
        if message["type"] != "submit_prompt":
            self.publish("protocol_error", reason="unknown_message_type")
            return
        text = message.get("text")
        if (not isinstance(input_id, str) or not 1 <= len(input_id) <= 128
                or not input_id.isascii()
                or any(ord(char) < 33 or ord(char) > 126 for char in input_id)):
            self._reject(None, "invalid_input_id")
            return
        if not isinstance(text, str) or not text.strip():
            self._reject(input_id, "empty_or_invalid_text")
            return
        try:
            encoded_text = text.encode("utf-8")
        except UnicodeError:
            self._reject(input_id, "invalid_text_encoding")
            return
        if len(encoded_text) > self.limits.prompt_bytes:
            self._reject(input_id, "prompt_too_large")
            return
        digest = hashlib.sha256(encoded_text).hexdigest()
        record = self.inputs.get(input_id)
        if record is not None:
            if digest != record.digest:
                self._reject(input_id, "input_id_conflict")
            else:
                self.publish("prompt_accepted", input_id=input_id,
                             status=record.status, outcome=record.outcome,
                             duplicate=True)
            return
        if self.enqueue is None:
            self._reject(input_id, "input_not_supported")
            return
        if self.closing:
            self._reject(input_id, "session_closing")
            return
        if self.paused and text.strip() != "/bridge resume":
            self._reject(input_id, "remote_execution_paused")
            return
        pending = sum(record.status == "queued"
                      for record in self.inputs.values())
        if pending >= self.limits.pending_inputs:
            self._reject(input_id, "queue_full")
            return
        self.enqueue(Submission(text, "bridge", input_id))
        self.inputs[input_id] = InputRecord(digest)
        self.publish("prompt_accepted", input_id=input_id, status="queued",
                     duplicate=False)

    def input_started(self, submission):
        self._status(submission, "running")

    def turn_started(self, submission, *, model=None, provider=None):
        turn_id = str(uuid.uuid4())
        self.active = {"turn_id": turn_id, "input_id": submission.input_id,
                       "origin": submission.origin, "model": model,
                       "provider": provider}
        self._status(submission, "running")
        self.publish("turn_started", **self.active)
        self._text(submission, submission.text, turn_id=turn_id,
                   event_type="input_chunk")
        return turn_id

    def _status(self, submission, status, outcome=None):
        record = self.inputs.get(submission.input_id)
        if submission.origin == "bridge" and record is not None:
            record.status = status
            record.outcome = outcome
        completed = [key for key, value in self.inputs.items()
                     if value.status == "finished"]
        excess = max(0, len(completed) - self.limits.completed_inputs)
        for key in completed[:excess]:
            del self.inputs[key]

    def _text(self, submission, text, *, turn_id=None,
              event_type="output_chunk"):
        # Chunk by characters with a conservative frame bound (JSON escaping
        # costs at most 12 bytes per Unicode character). Never truncate output.
        chunk_size = min(4096, max(1, (self.limits.frame_bytes - 1024) // 12))
        for index, offset in enumerate(range(0, len(text), chunk_size)):
            self.publish(event_type, input_id=submission.input_id,
                         origin=submission.origin, turn_id=turn_id,
                         chunk_index=index, text=text[offset:offset + chunk_size])

    def command_finished(self, submission, outcome, text):
        self._text(submission, text)
        self._status(submission, "finished", outcome)
        self.publish("command_finished", input_id=submission.input_id,
                     origin=submission.origin, outcome=outcome)

    def turn_finished(self, submission, turn_id, outcome, text):
        self._text(submission, text, turn_id=turn_id)
        self._status(submission, "finished", outcome)
        self.active = None
        if outcome == "cancelled":
            self.paused = True
        self.publish("turn_finished", input_id=submission.input_id,
                     origin=submission.origin, turn_id=turn_id,
                     outcome=outcome, paused=self.paused)

    def resume(self):
        self.paused = False
        self.publish("session_state", paused=False)
