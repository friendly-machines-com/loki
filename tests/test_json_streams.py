"""Bounded, selective JSON reading independent of filesystem names."""

import io
import json
import random
import unittest
from contextlib import aclosing
from unittest import mock

from loki_agent import json_streams


class Fragments:
    """A short-reading text source; an exhausted guarded source must not be read."""

    def __init__(self, pieces, *, guarded=False):
        self.pieces = iter(pieces)
        self.pending = ''
        self.guarded = guarded
        self.reads = []
        self.characters = 0
        self.closed = False

    def read(self, size=-1):
        if not 0 < size <= json_streams.CHUNK_SIZE:
            raise AssertionError('reader requested an unbounded/oversized read')
        self.reads.append(size)
        while not self.pending:
            self.pending = next(self.pieces, None)
            if self.pending is None:
                if self.guarded:
                    raise AssertionError('reader consumed the unnecessary tail')
                self.pending = ''
                return ''
        part = self.pending[:size]
        self.pending = self.pending[len(part):]
        self.characters += len(part)
        return part

    def close(self):
        self.closed = True


class JSONStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_strings_match_stdlib_at_every_split(self):
        values = ['plain', 'quote " and slash \\ /', '\b\f\n\r\t', 'snow \u96ea and emoji \U0001f600',
                  '\ud800 alone', '\udc00', '\ud800\udc00', '']
        for value in values:
            for ascii_only in [False, True]:
                wire = json.dumps(value, ensure_ascii=ascii_only)
                expected = json.loads(wire)
                for split in range(len(wire) + 1):
                    with self.subTest(value=repr(value), ascii=ascii_only, split=split):
                        stream = Fragments([wire[:split], wire[split:]])
                        reader = json_streams.Reader(stream, chunk_size=7)
                        parts = []
                        async with aclosing(reader.string_chunks()) as chunks:
                            async for part in chunks:
                                parts.append(part)
                        self.assertEqual(''.join(parts), expected)
                        self.assertFalse(stream.closed, 'reader borrows rather than owns the stream')

    async def test_small_scalar_capture_decodes_escaped_keys(self):
        stream = Fragments(['{"\\u0065vents":', '[] , "', 'x' * 2000, '": null}'])
        reader = json_streams.Reader(stream, chunk_size=11)
        await reader.expect('{')
        self.assertEqual(await reader.key(first=True), 'events')
        await reader.skip()
        self.assertEqual(await reader.key(first=False), '')
        await reader.skip()
        self.assertIsNone(await reader.key(first=False))
        self.assertEqual(await reader.kind(), '')

    async def test_large_strings_are_skipped_without_materializing_a_scalar(self):
        def pieces():
            yield '"'
            for _ in range(4096):
                yield 'x' * 4096
            yield '"'
        reader = json_streams.Reader(Fragments(pieces()), chunk_size=4096)
        with mock.patch.object(json_streams.json, 'loads', side_effect=AssertionError('large scalar decoded')):
            self.assertIsNone(await reader.read_string(8))

    async def test_skips_each_value_and_leaves_the_next_delimiter(self):
        values = [None, True, False, 0, -0.25, 12e30, -4e-12, float('nan'), float('inf'),
                  -float('inf'), 'quoted \\" [ braces }', [], {},
                  {'key': [1, None, {'nested': 'escaped \n \U0001f600'}], 'other': False}]
        for value in values:
            wire = json.dumps(value) + ' ,'
            for size in [1, 2, 7, 4096]:
                with self.subTest(value=repr(value), chunk_size=size):
                    reader = json_streams.Reader(io.StringIO(wire), chunk_size=size)
                    await reader.skip()
                    self.assertEqual(await reader.kind(), ',')

    async def test_deterministic_generated_values_leave_exactly_the_following_value(self):
        generator = random.Random(712)
        atoms = [None, True, False, 0, -3.5, 12e30, '', 'quote " slash \\ newline\n', '\u96ea \U0001f600']

        def value(depth):
            if depth == 0 or generator.randrange(3) == 0:
                return generator.choice(atoms)
            if generator.randrange(2):
                return [value(depth - 1) for _ in range(generator.randrange(4))]
            return {str(index) + '\u96ea"': value(depth - 1) for index in range(generator.randrange(4))}

        for index in range(100):
            first = json.dumps(value(4), ensure_ascii=bool(index % 2))
            reader = json_streams.Reader(io.StringIO('[' + first + ', "following"]'), chunk_size=1 + index % 13)
            await reader.expect('[')
            self.assertTrue(await reader.item(first=True))
            await reader.skip()
            self.assertTrue(await reader.item(first=False))
            self.assertEqual(await reader.read_string(20), 'following')
            self.assertFalse(await reader.item(first=False))

    async def test_deep_containers_do_not_use_python_recursion(self):
        reader = json_streams.Reader(io.StringIO('[' * 2500 + 'null' + ']' * 2500), chunk_size=31)
        await reader.skip()
        self.assertEqual(await reader.kind(), '')

    async def test_malformed_and_incomplete_prefixes_are_reported(self):
        values = ['{"x":}', '[1,]', '{"x":1,}', '"\\q"', '"\\u12XZ"', '"line\nbreak"',
                  '01', '1.', '-', 'falsex', 'tru', '[', '{', '1e', '1e+', '"unterminated', '"\\uD8']
        for value in values:
            with self.subTest(value=value):
                reader = json_streams.Reader(io.StringIO(value), chunk_size=1)
                with self.assertRaises(json_streams.Error):
                    await reader.skip()

    async def test_capture_limit_is_on_decoded_characters(self):
        for wire in ['"abcd"', '"\\u0061bcd"', '"\U0001f600bcd"']:
            for limit, expected in [[3, None], [4, json.loads(wire)]]:
                with self.subTest(wire=wire, limit=limit):
                    reader = json_streams.Reader(io.StringIO(wire), chunk_size=1)
                    self.assertEqual(await reader.read_string(limit), expected)
