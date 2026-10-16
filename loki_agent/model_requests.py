"""Model request execution, streaming, retries, and response interpretation.

The runtime captures request inputs before awaiting. This module owns no global
Session, active provider, tool registry, or transcript-persistence policy.
"""

import asyncio
import copy
import json
import logging
import os
import random
import shutil
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pprint import pformat

from . import authentications, connections, formats, http_client, protocols, sse
from . import response_headers as header_observations


@dataclass(frozen=True)
class RequestContext:
    """Authority, connection and identity selected for one sampling operation."""

    config: connections.RuntimeConfig | None
    credential_authority: authentications.CredentialAuthority | None
    conversation_id: str
    root_conversation_id: str
    response_headers: header_observations.Store


LLM_REQUEST_TIMEOUT_S = 300
LLM_STREAM_IDLE_TIMEOUT_S = 300

HTTP_RETRY_MAX_ATTEMPTS_LLM = 3     # for chat-completion POSTs (with idempotency key)
# Idempotency-key header injected on every chat-completion POST so the provider
# can dedup a retry. Anthropic honors this server-side; OpenAI-compat servers
# may ignore it and bill for each attempt -- set HTTP_RETRY_MAX_ATTEMPTS_LLM=1
# if your provider does not honor the header.
LLM_IDEMPOTENCY_HEADER_ANTHROPIC = "anthropic-idempotency-key"
LLM_IDEMPOTENCY_HEADER_OPENAI = "Idempotency-Key"
CODEX_TURN_STATE_HEADER = "x-codex-turn-state"
OPENCODE_SESSION_HEADER = "x-opencode-session"


@dataclass(frozen=True)
class TurnThinkingSettings:
    """The thinking settings of one logical turn, captured before any
    inference starts.

    Every request in the turn -- terminal inference, tool continuations,
    tool-less helpers, delegated inference -- uses this same snapshot, so
    changes made mid-turn apply to the next turn, never halfway through
    this one.
    """

    effort: str | None = None
    mode: str | None = None
    budget: int | None = None
    retention: str = "default"
    traces: str = "off"


@dataclass
class CodexTurnState:
    """Opaque sticky-routing state for one logical Codex turn.

    The ChatGPT backend mints this value on the first successful response.
    Every later sampling request in the same user turn must echo that exact
    first value. It is deliberately neither interpreted nor persisted.
    """

    value: str | None = None

    def capture(self, value):
        # Match Codex's OnceLock semantics: absence does nothing and the first
        # actual header value wins, even if a later response supplies another.
        if self.value is None and isinstance(value, str):
            self.value = value


class ApiError(Exception):
    def __init__(self, request_url: str, status: int, reason: str, body_text: str):
        self.request_url = request_url
        self.status = status
        self.reason = reason
        self.body_text = body_text
        try:
            self.body = json.loads(body_text) if body_text else None
        except json.JSONDecodeError:
            self.body = None
        super().__init__(self.summary())

    def summary(self) -> str:
        return f"API Error for <{self.request_url}>: HTTP {self.status} {self.reason}"

    def formatted_body(self) -> str:
        if self.body is not None:
            # API error bodies are often JSON dictionaries. Pretty-print them so
            # terminal errors stay readable. Use the real terminal width instead
            # of an arbitrary fixed column so wide terminals are not wrapped early.
            width = shutil.get_terminal_size(fallback=(80, 24)).columns
            return pformat(self.body, width=width)
        return self.body_text

    def formatted(self) -> str:
        body = self.formatted_body()
        if not body:
            return self.summary()
        return f"{self.summary()}:\n{body}"


class StreamingApiError(ApiError):
    def formatted(self) -> str:
        return (
            super().formatted()
            + "\nStreaming was requested. If this server does not support "
              "streaming, set LOKI_STREAM=0.")


class StreamCancelled(Exception):
    pass


def _codex_turn_state_for_request(
        config, request_url, supplied_state=None):
    """Return routing state only for the authenticated Codex Responses URL."""
    if (config is None
            or config.chat_provider.kind != protocols.OPENAI_RESPONSES
            or config.auth_spec is None
            or config.auth_spec.scheme != "openai-subscription"
            or request_url != config.chat_provider.chat_url):
        return None
    if supplied_state is not None:
        return supplied_state
    # A direct one-off request still needs stable state across its own
    # transport retries, even though it has no enclosing tool loop.
    return CodexTurnState()


def _prepare_codex_session_headers(headers, session):
    # Codex 0.144.1 uses persistent agent-tree identity for session-id, and
    # individual thread identity for both thread-id and x-client-request-id.
    # These are not per-request UUIDs or the ephemeral turn-state token.
    owned = {
        "session-id": session.root_conversation_id,
        "thread-id": session.conversation_id,
        "x-client-request-id": session.conversation_id,
    }
    for name in list(headers):
        if str(name).lower() in owned:
            del headers[name]
    headers.update(owned)


def _prepare_codex_turn_headers(headers, turn_state):
    if turn_state is None:
        return
    # Do not accept a caller-supplied or differently-cased stale value. The
    # only valid token is the first value captured in this logical turn.
    for name in list(headers):
        if str(name).lower() == CODEX_TURN_STATE_HEADER:
            del headers[name]
    if turn_state.value is not None:
        headers[CODEX_TURN_STATE_HEADER] = turn_state.value


def _capture_codex_turn_state(headers, turn_state):
    if turn_state is None or not isinstance(headers, dict):
        return
    for name, value in headers.items():
        if str(name).lower() == CODEX_TURN_STATE_HEADER:
            turn_state.capture(value)
            return


_OPENCODE_GO_INFERENCE_PATHS = frozenset({
    # Legacy Zen Go inference paths; retained so existing configurations
    # keep receiving a conversation identity.
    "/zen/go/v1/chat/completions",
    "/zen/go/v1/messages",
    "/zen/go/v1/responses",
    # Current OpenCode Go inference paths (see opencode.ai/docs/go).
    "/inference/go/openai/v1/chat/completions",
    "/inference/go/anthropic/v1/messages",
    "/inference/go/openai/v1/responses",
})


def _opencode_session_id_for_request(
        config, request_url, supplied_session_id=None):
    """Return conversation state only for canonical OpenCode Go inference."""
    if (config is None
            or request_url != config.chat_provider.chat_url):
        return None
    parsed = urllib.parse.urlparse(request_url)
    try:
        canonical_target = (
            parsed.scheme.lower() == "https"
            and parsed.hostname == "opencode.ai"
            and parsed.port in [None, 443]
            and parsed.username is None
            and parsed.password is None
            and parsed.path.rstrip("/") in _OPENCODE_GO_INFERENCE_PATHS
        )
    except ValueError:
        canonical_target = False
    if not canonical_target:
        return None
    if (not isinstance(supplied_session_id, str)
            or not supplied_session_id):
        raise ValueError(
            "OpenCode Go requests require a conversation identity")
    return supplied_session_id


def _prepare_opencode_session_headers(headers, session_id):
    if session_id is None:
        return
    # Provider headers and saved connection data are not authority to choose
    # conversation identity. Remove any casing variant before installing the
    # value owned by the current Session.
    for name in list(headers):
        if str(name).lower() == OPENCODE_SESSION_HEADER:
            del headers[name]
    headers[OPENCODE_SESSION_HEADER] = session_id


def _chat_response_observer(config, request_url, *, context):
    if config is None or request_url != config.chat_provider.chat_url:
        return None
    provider = config.chat_provider
    credential = config.auth_spec.credential if config.auth_spec else None
    return context.response_headers.observer(
        request_url, credential.encode() if credential else None,
        config.model, provider.provider_id, provider.provider_name)


async def async_provider_request(
        method: str, request_url: str, payload=None,
        request_headers: dict | None = None, report_errors: bool = False,
        show_timing: bool = False, cancel_check=None,
        codex_turn_state=None,
        opencode_session_id=None, *, context: RequestContext) -> protocols.ProviderResponse:
    start = time.perf_counter()
    config = context.config
    method = method.upper()
    observe = (_chat_response_observer(config, request_url, context=context)
               if method == "POST" else None)
    if method not in {"GET", "POST"}:
        raise ValueError("provider request method must be GET or POST")
    if method == "GET" and payload is not None:
        raise ValueError("GET provider requests cannot contain a payload")
    if method == "POST" and payload is None:
        raise ValueError("POST provider requests require a payload")
    body = (
        json.dumps(payload).encode("utf-8")
        if method == "POST" else b"")
    turn_state = (
        _codex_turn_state_for_request(
            config, request_url, codex_turn_state)
        if method == 'POST' else None)
    opencode_session_id = (
        _opencode_session_id_for_request(
            config, request_url, opencode_session_id)
        if method == 'POST' else None)

    base_headers = request_headers
    if base_headers is None:
        base_headers = (
            config.chat_provider.headers if config else {})
    # Copy so the per-call idempotency key below does not mutate the cached
    # provider headers shared across requests.
    base_headers = dict(base_headers)
    if turn_state is not None:
        _prepare_codex_session_headers(base_headers, context)

    retry_attempts = http_client.HTTP_RETRY_MAX_ATTEMPTS
    request_timeout = http_client.HTTP_DEFAULT_TIMEOUT_S
    if method == 'POST':
        # Best-effort server-side dedup; Anthropic honors this header on
        # /v1/messages. OpenAI-compat servers may ignore it.
        kind = config.chat_provider.kind if config else None
        header = (LLM_IDEMPOTENCY_HEADER_ANTHROPIC
                  if kind == protocols.ANTHROPIC_MESSAGES
                  else LLM_IDEMPOTENCY_HEADER_OPENAI)
        base_headers[header] = uuid.uuid4().hex
        retry_attempts = HTTP_RETRY_MAX_ATTEMPTS_LLM
        request_timeout = LLM_REQUEST_TIMEOUT_S

    rejected_generation = None
    recovered = False
    while True:
        headers_to_use, lease = (
            await authentications.authorized_request_headers(
                context.credential_authority,
                config.auth_spec if config is not None else None,
                request_url,
                base_headers,
                rejected_generation,
            ))
        _prepare_opencode_session_headers(
            headers_to_use, opencode_session_id)
        transport_options = {
            "body": body,
            "headers_in": headers_to_use,
            "timeout": request_timeout,
            "max_bytes": http_client.HTTP_MAX_RESPONSE_BYTES,
            "retry_max_attempts": retry_attempts,
            "retry_base_delay_s": http_client.HTTP_RETRY_BASE_DELAY_S,
            "retry_max_jitter_s": http_client.HTTP_RETRY_MAX_JITTER_S,
            "retry_backoff_factor": http_client.HTTP_RETRY_BACKOFF_FACTOR,
            "cancel_check": cancel_check,
        }

        def on_response_headers(status, headers):
            if observe is not None:
                observe(status, headers)
            if 200 <= status < 300:
                _capture_codex_turn_state(headers, turn_state)

        if observe is not None or turn_state is not None:
            transport_options["on_response_headers"] = on_response_headers
        if turn_state is not None:
            transport_options["prepare_attempt_headers"] = (
                lambda headers: _prepare_codex_turn_headers(headers, turn_state))
        response = await http_client.async_http_request(
            method,
            request_url,
            **transport_options,
        )
        if (response.status == 401
                and lease is not None
                and lease.refreshable
                and not recovered):
            # The generation check in the broker ensures concurrent stale
            # 401s observe an already-refreshed token instead of replaying
            # the rotating refresh token.
            rejected_generation = lease.generation
            recovered = True
            continue
        break
    response_text = http_client.decode_text(response.body, response.headers)
    if response.status >= 400:
        raise ApiError(request_url, response.status, response.reason, response_text)
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise protocols.ProtocolError(
            f"response from <{request_url}> was not valid JSON: "
            f"{error.msg}",
            payload=response_text,
        ) from error

    elapsed = time.perf_counter() - start
    if show_timing:
        logging.getLogger(__name__).debug(
            "LLM Response Time: %.3fs", elapsed)
    return protocols.ProviderResponse(
        data,
        effective_model=(
            protocols.openai_response_model_header(response.headers)
            if turn_state is not None else None),
    )


async def _next_stream_chunk(iterator, cancel_check):
    if cancel_check():
        raise StreamCancelled()
    task = asyncio.ensure_future(anext(iterator))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.05)
            if done:
                return task.result()
            if cancel_check():
                raise StreamCancelled()
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


async def _collect_stream_body(iterator, first_chunk, cancel_check):
    chunks = [first_chunk] if first_chunk else []
    while True:
        try:
            chunk = await _next_stream_chunk(iterator, cancel_check)
        except StopAsyncIteration:
            return b"".join(chunks)
        chunks.append(chunk)


def _stream_body_kind(content_type, first_chunk):
    # Leading whitespace is legal before a JSON document but NOT before an
    # SSE field (SseDecoder would drop such a line), so strip it only for
    # the JSON check: "   data:" classifies as JSON and fails loudly, not
    # as SSE that silently loses its first event.
    prefix = first_chunk.lstrip()
    if prefix.startswith((b"{", b"[")):
        return "json"
    # A UTF-8 BOM may precede the first SSE line; SseDecoder strips one
    # per line itself (sse.py), so the sniffer must accept it here too.
    if first_chunk.startswith(b"\xef\xbb\xbf"):
        if first_chunk[3:].lstrip().startswith(
                (b"data:", b"event:", b":")):
            return "sse"
    if first_chunk.startswith((b"data:", b"event:", b":")):
        return "sse"
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type == "text/event-stream":
        return "sse"
    if media_type in ["application/json", "application/problem+json"]:
        return "json"
    return "json"


_STREAM_MARKERS = (b"data:", b"event:", b":")


def _prefix_ambiguous(prefix):
    """Whether prefix so far could still become an SSE field marker.

    A first TCP read can end mid-marker ("da", "data") or hold only
    whitespace; such prefixes must not be classified yet, and one more
    chunk resolves them. Whitespace alone stays ambiguous even though a
    leading-whitespace stream cannot be SSE, because the whitespace may be
    the padding of a JSON document whose "{" arrives in the next chunk.
    """
    stripped = prefix.lstrip()
    if not stripped:
        return True
    if stripped.startswith(b"\xef\xbb\xbf"):
        stripped = stripped[3:]
        if not stripped:
            return True
    return any(marker.startswith(stripped) for marker in _STREAM_MARKERS)


async def _first_body_chunk(iterator, cancel_check):
    """Read body bytes until the SSE-vs-JSON sniff can decide.

    Returns the concatenated bytes (b"" on immediate EOF). Extra reads
    happen only while the bytes so far are still an ambiguous prefix of an
    SSE marker, so a normal first line (or JSON body) never blocks on this
    bounded lookahead.
    """
    chunks = []
    total = 0
    while True:
        try:
            chunk = await _next_stream_chunk(iterator, cancel_check)
        except StopAsyncIteration:
            return b"".join(chunks)
        if chunk:
            chunks.append(chunk)
            total += len(chunk)
        if not _prefix_ambiguous(b"".join(chunks)[-64:]):
            return b"".join(chunks)


async def _async_chat_stream_request_once(
        request_url, payload, request_headers, on_text_delta, cancel_check,
        codex_turn_state=None, observe=None, on_reasoning_delta=None,
        *, config) -> protocols.ProviderResponse:
    body = json.dumps(payload).encode("utf-8")
    async with http_client.async_http_stream(
            "POST",
            request_url,
            body=body,
            headers_in=request_headers,
            timeout=LLM_STREAM_IDLE_TIMEOUT_S,
            max_bytes=http_client.HTTP_MAX_RESPONSE_BYTES,
            cancel_check=cancel_check) as response:
        if observe is not None:
            observe(response.status, response.headers)
        effective_model = (
            protocols.openai_response_model_header(response.headers)
            if codex_turn_state is not None else None)
        if 200 <= response.status < 300:
            # Capture as soon as the HTTP headers arrive. If the body then
            # fails before yielding an SSE event, the transport retry must
            # already be able to replay the routing token.
            _capture_codex_turn_state(
                response.headers, codex_turn_state)
        iterator = response.body.__aiter__()
        first_chunk = await _first_body_chunk(iterator, cancel_check)

        if response.status >= 400:
            try:
                response_body = await _collect_stream_body(
                    iterator, first_chunk, cancel_check)
            except OSError as e:
                if first_chunk:
                    raise protocols.StreamProtocolError(
                        f"stream interrupted after output began: {e}") from e
                raise
            response_text = http_client.decode_text(
                response_body, response.headers)
            raise StreamingApiError(
                request_url, response.status, response.reason, response_text)

        if _stream_body_kind(
                response.header("content-type"), first_chunk) == "json":
            try:
                response_body = await _collect_stream_body(
                    iterator, first_chunk, cancel_check)
            except OSError as e:
                if first_chunk:
                    raise protocols.StreamProtocolError(
                        f"stream interrupted after output began: {e}") from e
                raise
            try:
                return protocols.ProviderResponse(
                    json.loads(http_client.decode_text(
                        response_body, response.headers)),
                    effective_model=effective_model,
                )
            except json.JSONDecodeError as e:
                raise protocols.StreamProtocolError(
                    "streaming response was neither valid SSE nor JSON: "
                    f"{e}") from e

        accumulator = config.chat_provider.stream_accumulator(
            on_text_delta, on_reasoning_delta)

        def observed_model():
            return (
                effective_model
                or getattr(accumulator, "effective_model", None))

        decoder = sse.SseDecoder()
        received_body = bool(first_chunk)
        try:
            for event in decoder.feed(first_chunk):
                if accumulator.feed(event):
                    return protocols.ProviderResponse(
                        accumulator.finish(),
                        effective_model=observed_model(),
                        reasoning_field=getattr(accumulator, "reasoning_field", None),
                        notice_codes=tuple(
                            getattr(accumulator, "notice_codes", ())),
                    )
            while True:
                try:
                    chunk = await _next_stream_chunk(
                        iterator, cancel_check)
                except StopAsyncIteration:
                    break
                received_body = received_body or bool(chunk)
                for event in decoder.feed(chunk):
                    if accumulator.feed(event):
                        return protocols.ProviderResponse(
                            accumulator.finish(),
                            effective_model=observed_model(),
                            reasoning_field=getattr(accumulator, "reasoning_field", None),
                            notice_codes=tuple(
                                getattr(
                                    accumulator, "notice_codes", ())),
                        )
            for event in decoder.finish():
                if accumulator.feed(event):
                    return protocols.ProviderResponse(
                        accumulator.finish(),
                        effective_model=observed_model(),
                        reasoning_field=getattr(accumulator, "reasoning_field", None),
                        notice_codes=tuple(
                            getattr(accumulator, "notice_codes", ())),
                    )
        except StreamCancelled:
            raise
        except protocols.ResponseApiError:
            raise
        except protocols.StreamProtocolError:
            raise
        except ValueError as e:
            raise protocols.StreamProtocolError(
                f"invalid SSE stream: {e}") from e
        except OSError as e:
            if received_body:
                raise protocols.StreamProtocolError(
                    f"stream interrupted after output began: {e}") from e
            raise
        return protocols.ProviderResponse(
            accumulator.finish(),
            effective_model=observed_model(),
            reasoning_field=getattr(accumulator, "reasoning_field", None),
            notice_codes=tuple(
                getattr(accumulator, "notice_codes", ())),
        )


async def async_chat_stream_request(
        request_url: str, payload, request_headers: dict | None = None,
        on_text_delta=None, cancel_check=None,
        report_errors: bool = False, show_timing: bool = False,
        codex_turn_state=None,
        opencode_session_id=None, on_reasoning_delta=None,
        *, context: RequestContext) -> protocols.ProviderResponse:
    start = time.perf_counter()
    config = context.config
    observe = _chat_response_observer(config, request_url, context=context)
    output_started = False

    def callback(text):
        nonlocal output_started
        output_started = output_started or bool(text)
        if on_text_delta is not None:
            on_text_delta(text)

    def reasoning_callback(key, kind, text):
        nonlocal output_started
        output_started = output_started or bool(text)
        if on_reasoning_delta is not None:
            on_reasoning_delta(key, kind, text)

    cancel = cancel_check or (lambda: False)
    turn_state = _codex_turn_state_for_request(
        config, request_url, codex_turn_state)
    opencode_session_id = _opencode_session_id_for_request(
        config, request_url, opencode_session_id)
    base_headers = dict(
        request_headers if request_headers is not None
        else (config.chat_provider.headers if config else {}))
    if turn_state is not None:
        _prepare_codex_session_headers(base_headers, context)
    base_headers.setdefault("Accept", "text/event-stream")
    kind = config.chat_provider.kind if config else None
    idempotency_header = (
        LLM_IDEMPOTENCY_HEADER_ANTHROPIC
        if kind == protocols.ANTHROPIC_MESSAGES
        else LLM_IDEMPOTENCY_HEADER_OPENAI)
    base_headers[idempotency_header] = uuid.uuid4().hex

    transport_attempt = 0
    rejected_generation = None
    recovered = False
    while True:
        headers_to_use, lease = (
            await authentications.authorized_request_headers(
                context.credential_authority,
                config.auth_spec if config is not None else None,
                request_url,
                base_headers,
                rejected_generation,
            ))
        _prepare_opencode_session_headers(
            headers_to_use, opencode_session_id)
        _prepare_codex_turn_headers(headers_to_use, turn_state)
        transport_attempt += 1
        try:
            stream_kwargs = {"codex_turn_state": turn_state, "observe": observe, "config": config}
            if on_reasoning_delta is not None:
                stream_kwargs["on_reasoning_delta"] = reasoning_callback
            response = await _async_chat_stream_request_once(
                request_url, payload, headers_to_use, callback, cancel, **stream_kwargs)
            break
        except http_client.HttpRequestCancelled:
            raise StreamCancelled()
        except Exception as exc:
            if (isinstance(exc, StreamingApiError)
                    and exc.status == 401
                    and lease is not None
                    and lease.refreshable
                    and not recovered):
                rejected_generation = lease.generation
                recovered = True
                # Authentication recovery is independent of transport retry
                # accounting and preserves the same idempotency key.
                transport_attempt = 0
                continue
            retryable_response_error = (
                isinstance(exc, protocols.ResponseApiError)
                and exc.retryable)
            if (output_started or transport_attempt >= HTTP_RETRY_MAX_ATTEMPTS_LLM
                    or not (
                        retryable_response_error
                        or http_client.is_transient_error(exc))):
                raise
            delay = (
                exc.retry_after
                if (retryable_response_error
                    and exc.retry_after is not None)
                else (
                    http_client.HTTP_RETRY_BASE_DELAY_S
                    * (http_client.HTTP_RETRY_BACKOFF_FACTOR
                       ** (transport_attempt - 1))))
            if (http_client.HTTP_RETRY_MAX_JITTER_S
                    and not (
                        retryable_response_error
                        and exc.retry_after is not None)):
                delay += random.uniform(0, http_client.HTTP_RETRY_MAX_JITTER_S)
            try:
                await http_client._wait_with_cancel(
                    asyncio.sleep(delay), delay + 0.1, cancel)
            except http_client.HttpRequestCancelled:
                raise StreamCancelled()

    elapsed = time.perf_counter() - start
    if show_timing:
        logging.getLogger(__name__).debug(
            "LLM Response Time: %.3fs", elapsed)
    return response


async def async_chat_completion(transcript_items: list, tools, report_errors: bool = False,
                                show_timing: bool = False,
                                on_text_delta=None, on_reasoning_delta=None,
                                cancel_check=None, codex_turn_state=None,
                                reasoning_effort=None, *, context: RequestContext,
                                thinking: TurnThinkingSettings) -> formats.DecodedTurn:
    if (reasoning_effort is not None
            and (context.config.reasoning_effort_profile is None
                 or not context.config.reasoning_effort_profile.supports(
                     reasoning_effort))):
        raise protocols.ProtocolError(
            "reasoning effort is not supported by the selected model")
    effective_model = context.config.model
    notice_codes = ()
    reasoning_source = context.config.chat_provider.reasoning_field

    if context.config.chat_provider.kind == protocols.DUMMY:
        # No-op LLM for testing: never touches the network.  The reply is a
        # canned assistant message, so the whole input/turn/render loop runs
        # deterministically. LOKI_DUMMY_STREAM_CHUNKS optionally supplies a
        # JSON array of deltas. A test can make the provider pause after its
        # first delta by setting LOKI_DUMMY_STREAM_GATE to a path that the test
        # creates after observing the corresponding terminal output.
        reply = os.environ.get("LOKI_DUMMY_REPLY", "ok")
        raw_chunks = os.environ.get("LOKI_DUMMY_STREAM_CHUNKS")
        if context.config.stream and on_text_delta and raw_chunks is not None:
            try:
                chunks = json.loads(raw_chunks)
            except json.JSONDecodeError as error:
                raise protocols.ProtocolError(
                    f"LOKI_DUMMY_STREAM_CHUNKS is invalid JSON: {error}"
                ) from error
            if (not isinstance(chunks, list)
                    or any(not isinstance(chunk, str) for chunk in chunks)):
                raise protocols.ProtocolError(
                    "LOKI_DUMMY_STREAM_CHUNKS must be a JSON array "
                    "of strings")
            gate = os.environ.get("LOKI_DUMMY_STREAM_GATE")
            for index, chunk in enumerate(chunks):
                on_text_delta(chunk)
                if index == 0 and gate:
                    deadline = time.monotonic() + 10
                    while not os.path.exists(gate):
                        if cancel_check and cancel_check():
                            raise StreamCancelled()
                        if time.monotonic() >= deadline:
                            raise protocols.ProtocolError(
                                "timed out waiting for "
                                "LOKI_DUMMY_STREAM_GATE")
                        await asyncio.sleep(0.01)
        # A test hook that makes the canned reply a tool call instead of text,
        # so a test can drive the model/tool loop through a real tool execution
        # with the DUMMY provider.  The value is a JSON object with a ``name``
        # and an optional ``arguments`` object.  It fires only while the
        # transcript contains no tool call yet, so the loop's follow-up model
        # call gets the ordinary text reply instead of the call again.
        tool_call_spec = os.environ.get("LOKI_DUMMY_TOOL_CALL")
        if tool_call_spec and not formats.response_tool_calls(transcript_items):
            try:
                spec = json.loads(tool_call_spec)
            except json.JSONDecodeError as error:
                raise protocols.ProtocolError(
                    f"LOKI_DUMMY_TOOL_CALL is invalid JSON: {error}"
                ) from error
            if (not isinstance(spec, dict)
                    or not isinstance(spec.get("name"), str)):
                raise protocols.ProtocolError(
                    "LOKI_DUMMY_TOOL_CALL must be an object with a "
                    "string 'name'")
            arguments = spec.get("arguments", {})
            if not isinstance(arguments, dict):
                raise protocols.ProtocolError(
                    "LOKI_DUMMY_TOOL_CALL 'arguments' must be an object")
            return formats.DecodedTurn(
                [formats.tool_call_item(
                    "dummy-tool-call", spec["name"], arguments)],
                {
                    "protocol": "dummy",
                    "provider_id": "dummy",
                    "model": context.config.model,
                    "response": {},
                },
            )
        return formats.DecodedTurn(
            [formats.message_item("assistant", reply)],
            {
                "protocol": "dummy",
                "provider_id": "dummy",
                "model": context.config.model,
                "response": {},
            },
        )

    opencode_session_id = _opencode_session_id_for_request(
        context.config,
        context.config.chat_provider.chat_url,
        context.conversation_id,
    )
    if context.config.stream:
        payload = context.config.chat_provider.streaming_chat_payload(
            transcript_items,
            tools,
            context.config.model,
            prompt_cache_key=context.conversation_id,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking.mode,
            thinking_budget=thinking.budget,
            reasoning_retention=thinking.retention,
            reasoning_traces=thinking.traces,
        )
        request_kwargs = {
            "context": context,
            "request_headers":
                context.config.chat_provider.headers,
            "on_text_delta": on_text_delta,
            "cancel_check": cancel_check,
            "report_errors": report_errors,
            "show_timing": False,
        }
        if on_reasoning_delta is not None:
            request_kwargs["on_reasoning_delta"] = on_reasoning_delta
        if codex_turn_state is not None:
            request_kwargs["codex_turn_state"] = codex_turn_state
        if opencode_session_id is not None:
            request_kwargs["opencode_session_id"] = opencode_session_id
        response = await async_chat_stream_request(
            context.config.chat_provider.chat_url,
            payload,
            **request_kwargs,
        )
        data = response.payload
        effective_model = response.effective_model or effective_model
        notice_codes = response.notice_codes
        reasoning_source = response.reasoning_field or reasoning_source
    else:
        payload = context.config.chat_provider.chat_payload(
            transcript_items,
            tools,
            context.config.model,
            prompt_cache_key=context.conversation_id,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking.mode,
            thinking_budget=thinking.budget,
            reasoning_retention=thinking.retention,
            reasoning_traces=thinking.traces,
        )
        try:
            request_kwargs = {
                "context": context,
                "request_headers":
                    context.config.chat_provider.headers,
                "report_errors": report_errors,
                "show_timing": show_timing,
            }
            if cancel_check is not None:
                request_kwargs["cancel_check"] = cancel_check
            if codex_turn_state is not None:
                request_kwargs["codex_turn_state"] = codex_turn_state
            if opencode_session_id is not None:
                request_kwargs["opencode_session_id"] = opencode_session_id
            response = await async_provider_request(
                "POST",
                context.config.chat_provider.chat_url,
                payload,
                **request_kwargs,
            )
            data = response.payload
            effective_model = response.effective_model or effective_model
            notice_codes = response.notice_codes
        except http_client.HttpRequestCancelled:
            raise StreamCancelled()
    if not isinstance(data, dict):
        raise protocols.ProtocolError(
            "chat response must be a JSON object", payload=data)
    detected = protocols.detect_protocol_from_response(data)
    if detected and detected != context.config.chat_provider.kind:
        # A configured adapter should not parse a response that clearly has
        # another protocol's shape; that usually means endpoint/config mismatch.
        raise protocols.ProtocolError(
            f"configured provider {context.config.chat_provider.kind!r} "
            f"but response looks like {detected!r}",
            payload=data,
        )
    try:
        turn = formats.coerce_decoded_turn(
            context.config.chat_provider.parse_chat_response(data))
    except formats.TranscriptFormatError as e:
        raise protocols.ProtocolError(
            f"invalid {context.config.chat_provider.kind} response: {e}",
            payload=data,
        ) from e
    provider = context.config.chat_provider
    turn.metadata["provider_id"] = provider.provider_id
    turn.metadata["provider_name"] = provider.provider_name
    turn.metadata["endpoint"] = (
        provider.chat_url)
    turn.metadata["model"] = effective_model
    turn.metadata["requested_model"] = context.config.model
    turn.metadata["protocol"] = context.config.chat_provider.kind
    if reasoning_source in ["reasoning", "reasoning_content", "reasoning_details"]:
        native = copy.deepcopy(turn.metadata.get("protocol_data") or {})
        if not isinstance(native, dict):
            native = {}
        native.setdefault("loki", {})["reasoning_field"] = reasoning_source
        turn.metadata["protocol_data"] = native
    if notice_codes:
        protocol_data = copy.deepcopy(
            turn.metadata.get("protocol_data") or {})
        if not isinstance(protocol_data, dict):
            protocol_data = {}
        protocol_data.setdefault("loki", {})["provider_notices"] = list(notice_codes)
        turn.metadata["protocol_data"] = protocol_data
    return turn
