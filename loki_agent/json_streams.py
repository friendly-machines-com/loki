"""Selective JSON reading for shared session summaries, not another DOM decoder.

json.load/raw_decode require complete aggregate values. The shared session lister
instead needs a historical-user prefix and selected trailing metadata without a
history DOM. Reader borrows a text stream; ObjectSuffix locates the final object
on its underlying binary handle. Neither owns filenames, file descriptors, tasks,
or cached conversation state; buffers and traversal state belong to one lookup.
Regular-file reads use the standard buffered stream; yielding before each bounded
refill keeps cancellation/other asyncio work runnable without worker threads.
Only the inspected prefix is checked: stopping a consumer does not validate its
unread suffix. Public json.loads handles bounded scalar/escape decoding rather
than depending on CPython/PyPy's private scanner implementations.
"""

import asyncio
from contextlib import aclosing
import io
import json
import re


CHUNK_SIZE = 16 * 1024
FIELD_CHARS = 64

_STRING_EDGE = re.compile(r'["\\\x00-\x1f]')
_SPACE = re.compile(r'[ \t\r\n]*')
_DIGITS = re.compile(r'[0-9]+')
_SCALAR = re.compile(r'[^ \t\r\n,\]}]+')
_HEX = frozenset('0123456789abcdefABCDEF')
_BACK_STRUCTURE = re.compile(rb'["{}\[\]]')


class Error(ValueError):
    """The inspected JSON prefix is malformed or incomplete."""


class ByteRange:
    """Feed only a located metadata value to Reader through a UTF-8 decoder.

    A decoder on the whole file could read/validate beyond that selected value.
    This borrowed range supplies EOF at its boundary; only the enclosing checked
    file scope closes the handle. No second open or retained metadata copy.
    """

    def __init__(self, stream, region):
        self.stream = stream
        self.remaining = region.stop - region.start
        stream.seek(region.start)

    def read(self, size):
        if size < 0:
            raise ValueError('range reads must be bounded')
        data = self.stream.read(min(size, self.remaining)) if self.remaining else b''
        self.remaining -= len(data)
        return data


class ObjectSuffix:
    """Locate the final object-valued field, not search backwards through history.

    The shared saved-session lister needs the trailing session_state written by
    serialize_chat_log. Forward Reader cannot reach it without consuming events.
    Only inspect that final object: an array/string/scalar suffix is NOT a reason
    to traverse another root value (especially events). Metadata preceding events
    is instead handled by the lister's prefix lookup. Reader subsequently checks
    the located object's JSON; this cursor does not validate skipped history.

    Borrow one checked binary handle. Buffer/structural offsets are bounded by a
    chunk, plus nesting depth; no file ownership, retained cache or async task.
    Every bounded refill yields, including during very large metadata values.
    """

    def __init__(self, stream):
        self.stream = stream
        self.offset = stream.seek(0, io.SEEK_END)
        self.start = self.offset
        self.buffer = b''
        self.edges = []

    async def _fill(self):
        if not self.offset:
            return False
        if not self.start <= self.offset - 1 < self.start + len(self.buffer):
            await asyncio.sleep(0)
            end = self.offset
            self.start = max(0, end - CHUNK_SIZE)
            self.stream.seek(self.start)
            self.buffer = self.stream.read(end - self.start)
            if len(self.buffer) != end - self.start:
                raise Error('Saved JSON changed during metadata lookup')
            self.edges = [match.start() for match in _BACK_STRUCTURE.finditer(self.buffer)]
        return True

    async def _peek(self):
        return self.buffer[self.offset - self.start - 1] if await self._fill() else None

    async def _kind(self):
        while await self._peek() in [32, 9, 13, 10]:
            self.offset -= 1
        return await self._peek()

    async def _expect(self, char):
        if await self._kind() != char:
            raise Error('Unexpected saved JSON suffix delimiter')
        self.offset -= 1

    async def _string(self):
        await self._expect(34)
        while await self._fill():
            position = self.buffer.rfind(b'"', 0, self.offset - self.start)
            if position < 0:
                self.offset = self.start
                continue
            quote = self.start + position
            self.offset = quote
            slashes = 0
            while await self._peek() == 92:
                self.offset -= 1
                slashes += 1
            if not slashes % 2:
                return quote
        raise Error('Unterminated saved JSON suffix string')

    async def field(self, name):
        await self._expect(125)
        if await self._kind() != 125:
            return None
        stop = self.offset
        self.offset -= 1
        pending = [123]
        while pending:
            if not await self._fill():
                raise Error('Unbalanced saved JSON suffix')
            while self.edges and self.edges[-1] >= self.offset - self.start:
                self.edges.pop()
            if not self.edges:
                self.offset = self.start
                continue
            position = self.edges.pop()
            self.offset = self.start + position + 1
            char = self.buffer[position]
            if char == 34:
                await self._string()
                continue
            self.offset -= 1
            if char in [125, 93]:
                pending.append(123 if char == 125 else 91)
            elif char == pending[-1]:
                pending.pop()
            else:
                raise Error('Mismatched saved JSON suffix container')
        start = self.offset
        await self._expect(58)
        await self._kind()
        key_stop = self.offset
        key_start = await self._string()
        if await self._kind() not in [44, 123]:
            raise Error('Invalid saved JSON field boundary')
        if key_stop - key_start > FIELD_CHARS * 12 + 2:
            return None
        self.stream.seek(key_start)
        try:
            key = json.loads(self.stream.read(key_stop - key_start).decode('utf-8'))
        except (ValueError, UnicodeError) as error:
            raise Error('Invalid saved JSON metadata key') from error
        return range(start, stop) if key == name else None


class Reader:
    def __init__(self, stream, *, chunk_size=CHUNK_SIZE):
        if chunk_size <= 0:
            raise ValueError('chunk_size must be positive')
        self.stream = stream
        self.chunk_size = chunk_size
        self.buffer = ''
        self.position = 0
        self.eof = False

    async def _fill(self):
        if self.position < len(self.buffer):
            return True
        if self.eof:
            return False
        await asyncio.sleep(0)
        self.buffer = self.stream.read(self.chunk_size)
        self.position = 0
        self.eof = not self.buffer
        return not self.eof

    async def _peek(self):
        return self.buffer[self.position] if await self._fill() else ''

    async def _take(self):
        char = await self._peek()
        if not char:
            raise Error('Unexpected end of JSON')
        self.position += 1
        return char

    async def kind(self):
        """Inspect the next value/delimiter after JSON whitespace."""
        while await self._fill():
            self.position = _SPACE.match(self.buffer, self.position).end()
            if self.position < len(self.buffer):
                return self.buffer[self.position]
        return ''

    async def expect(self, char):
        if await self.kind() != char:
            raise Error('Unexpected JSON delimiter')
        self.position += 1

    async def _raw_string(self):
        await self.expect('"')
        while await self._fill():
            match = _STRING_EDGE.search(self.buffer, self.position)
            end = match.start() if match else len(self.buffer)
            if end > self.position:
                part = self.buffer[self.position:end]
                self.position = end
                yield part
            if match is None:
                continue
            char = await self._take()
            if char == '"':
                return
            if char != '\\':
                raise Error('Unescaped control character in JSON string')
            escape = await self._take()
            if escape == 'u':
                digits = ''.join([await self._take() for _ in range(4)])
                if any(digit not in _HEX for digit in digits):
                    raise Error('Invalid JSON Unicode escape')
                yield '\\u' + digits
            elif escape in '"\\/bfnrt':
                yield '\\' + escape
            else:
                raise Error('Invalid JSON escape')
        raise Error('Unterminated JSON string')

    async def read_string(self, max_chars):
        """Consume a string; return None when its decoded value exceeds the cap.

        Keys/type/role fields need equality checks, not arbitrary large strings.
        At most twelve encoded characters represent one decoded surrogate pair.
        Once that capture budget is exceeded, validate/skip the rest without
        retaining it. max_chars=0 is the same operation with no value capture.
        """
        parts = []
        length = 0
        overflow = False
        async for part in self._raw_string():
            length += len(part)
            if length > max_chars * 12:
                overflow = True
                parts.clear()
            if not overflow:
                parts.append(part)
        if overflow:
            return None
        value = json.loads('"' + ''.join(parts) + '"')
        return value if len(value) <= max_chars else None

    async def string_chunks(self):
        """Yield decoded string fragments without assembling the whole value.

        A pending escaped high surrogate belongs to this one string. Keep it
        until the next fragment establishes whether stdlib decoding should join
        a low surrogate; never join separate JSON strings or chunk boundaries
        incorrectly. Closing the iterator releases data only, not the stream.
        """
        high = ''
        async with aclosing(self._raw_string()) as chunks:
            async for part in chunks:
                if part.startswith('\\'):
                    value = json.loads('"' + part + '"')
                    code = ord(value)
                    if high and 0xDC00 <= code <= 0xDFFF and part.startswith('\\u'):
                        yield json.loads('"' + high + part + '"')
                        high = ''
                        continue
                    if high:
                        yield json.loads('"' + high + '"')
                        high = ''
                    if 0xD800 <= code <= 0xDBFF and part.startswith('\\u'):
                        high = part
                    else:
                        yield value
                else:
                    if high:
                        yield json.loads('"' + high + '"')
                        high = ''
                    yield part
        if high:
            yield json.loads('"' + high + '"')

    async def scalar(self):
        """Capture one requested scalar, not an object/array or ignored history.

        Saved cwd is an opaque lookup key: unlike preview/type fields it cannot
        be clipped to FIELD_CHARS. Its complete value (and the scalar itself)
        necessarily costs memory; unrelated strings/containers still use skip().
        Public JSON decoding also preserves the existing metadata fallback's
        numeric/null/boolean truthiness rather than guessing it from text.
        """
        kind = await self.kind()
        if kind in ['{', '[']:
            raise Error('Expected JSON scalar, not a container')
        if kind == '"':
            parts = []
            async for part in self.string_chunks():
                parts.append(part)
            return ''.join(parts)
        parts = []
        while await self._fill():
            match = _SCALAR.match(self.buffer, self.position)
            if match is None:
                break
            parts.append(match.group())
            self.position = match.end()
            if self.position < len(self.buffer):
                break
        try:
            return json.loads(''.join(parts))
        except ValueError as error:
            raise Error('Invalid requested JSON scalar') from error

    async def key(self, *, first, max_chars=FIELD_CHARS):
        """Read an object's next key after its opener; None consumes its end.

        The caller consumes exactly one value before requesting the next key.
        Oversized keys become an unrecognized empty key, not a retained token.
        No current selector needs an empty key or an unbounded property name.
        """
        if await self.kind() == '}':
            self.position += 1
            return None
        if not first:
            await self.expect(',')
        key = await self.read_string(max_chars)
        await self.expect(':')
        return key if key is not None else ''

    async def item(self, *, first):
        """Position at the next array value, or consume the array's end."""
        if await self.kind() == ']':
            self.position += 1
            return False
        if not first:
            await self.expect(',')
        return True

    async def _digits(self):
        found = False
        while await self._fill():
            match = _DIGITS.match(self.buffer, self.position)
            if match is None:
                break
            found = True
            self.position = match.end()
            if self.position < len(self.buffer):
                break
        if not found:
            raise Error('Expected JSON digit')

    async def _boundary(self):
        char = await self._peek()
        if char and char not in ' \t\r\n,]}':
            raise Error('Invalid JSON scalar')

    async def _literal(self, word):
        for char in word:
            if await self._take() != char:
                raise Error('Invalid JSON literal')
        await self._boundary()

    async def _number(self):
        if await self._peek() == '0':
            self.position += 1
        else:
            await self._digits()
        if await self._peek() == '.':
            self.position += 1
            await self._digits()
        if await self._peek() in ['e', 'E']:
            self.position += 1
            if await self._peek() in ['+', '-']:
                self.position += 1
            await self._digits()
        await self._boundary()

    async def skip(self):
        """Consume one unwanted value with O(nesting depth) structural state.

        Large strings, keys and numbers are never decoded into aggregate values.
        Accept the non-finite literals the existing stdlib loader/writer accept;
        the picker must not introduce a different saved-data authority policy.
        """
        pending = ['value']
        while pending:
            action = pending.pop()
            if action in ['object_first', 'object_next']:
                if await self.key(first=action == 'object_first', max_chars=0) is not None:
                    pending.extend(['object_next', 'value'])
                continue
            if action in ['array_first', 'array_next']:
                if await self.item(first=action == 'array_first'):
                    pending.extend(['array_next', 'value'])
                continue
            char = await self.kind()
            if char == '{':
                self.position += 1
                pending.append('object_first')
            elif char == '[':
                self.position += 1
                pending.append('array_first')
            elif char == '"':
                await self.read_string(0)
            elif char in ['t', 'f', 'n', 'N', 'I']:
                word = {'t': 'true', 'f': 'false', 'n': 'null', 'N': 'NaN', 'I': 'Infinity'}[char]
                await self._literal(word)
            elif char == '-':
                self.position += 1
                if await self._peek() == 'I':
                    await self._literal('Infinity')
                else:
                    await self._number()
            elif char and char in '0123456789':
                await self._number()
            else:
                raise Error('Expected JSON value')
