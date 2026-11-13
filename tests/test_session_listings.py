"""Session summaries must not consume the intervening conversation history."""

import asyncio
from contextlib import contextmanager
import io
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from loki_agent import formats, json_streams, savefiles


class MeasuredBytes(io.BytesIO):
    def __init__(self, value, *, guard=False):
        super().__init__(value)
        self.length = len(value)
        self.consumed = 0
        self.guard = guard

    def read(self, size=-1):
        if size < 0:
            raise AssertionError('listing read must be bounded')
        start = self.tell()
        stop = min(start + size, self.length)
        if self.guard and start < self.length - json_streams.CHUNK_SIZE and stop > json_streams.CHUNK_SIZE:
            raise AssertionError('listing read the intervening history')
        value = super().read(size)
        self.consumed += len(value)
        return value


class SessionSummaryTests(unittest.IsolatedAsyncioTestCase):
    @contextmanager
    def source(self, stream, *, name='chat-saved.json', modified=1000):
        entry = SimpleNamespace(name=name, path=object())

        @contextmanager
        def opening(path):
            self.assertIs(path, entry.path)
            try:
                yield [SimpleNamespace(buffer=stream), SimpleNamespace(st_mtime=modified)]
            finally:
                stream.close()

        with mock.patch.object(savefiles, 'chat_log_entries', return_value=[entry]), \
                mock.patch.object(savefiles, 'open_chat_log', side_effect=opening) as opened:
            yield [entry, opened]

    def wire(self, middle, *, before=False):
        first = json.dumps(formats.message_item('user', 'historical request'))
        state = json.dumps({'shell_cwd': 'raw/../cwd/./'})
        events = '"events":[' + first + ',"' + 'x' * middle + '"]'
        return ('{"session_state":' + state + ',' + events + '}' if before
                else '{' + events + ',"session_state":' + state + '}').encode()

    async def test_growing_history_does_not_grow_reads_in_either_metadata_layout(self):
        for before in [False, True]:
            consumed = []
            for size in [128 * 1024, 4 * 1024 * 1024]:
                source = MeasuredBytes(self.wire(size, before=before), guard=True)
                with self.subTest(before=before, size=size), self.source(source) as fixture, \
                        mock.patch.object(savefiles, 'os', SimpleNamespace(stat=mock.Mock(side_effect=AssertionError('pathname restat')))), \
                        mock.patch.object(savefiles.json, 'load', side_effect=AssertionError('history DOM')):
                    rows = await savefiles.list_saved_sessions('opaque-directory-key')
                    self.assertEqual(rows, [{'id': 'saved', 'path': fixture[0].path, 'mtime': 1000,
                                             'preview': 'historical request', 'cwd': 'raw/../cwd/./'}])
                    fixture[1].assert_called_once_with(fixture[0].path)
                self.assertTrue(source.closed)
                consumed.append(source.consumed)
            self.assertEqual(consumed[0], consumed[1])
            self.assertLess(consumed[0], 4 * json_streams.CHUNK_SIZE)

    async def test_missing_metadata_does_not_walk_the_history_or_build_a_preview(self):
        wire = b'{"events":["' + b'x' * (4 * 1024 * 1024) + b'"]}'
        source = MeasuredBytes(wire, guard=True)
        with self.source(source), mock.patch.object(savefiles, 'preview', side_effect=AssertionError('preview of rejected log')):
            self.assertEqual(await savefiles.list_saved_sessions('opaque'), [])
        self.assertLessEqual(source.consumed, 2 * json_streams.CHUNK_SIZE)
        self.assertTrue(source.closed)

    async def test_empty_label_is_not_a_rejected_session(self):
        source = MeasuredBytes(b'{"events":[],"session_state":{"cwd":"kept"}}')
        with self.source(source):
            rows = await savefiles.list_saved_sessions('opaque')
        self.assertEqual(rows[0]['preview'], '')
        self.assertEqual(rows[0]['cwd'], 'kept')

    async def test_summary_only_validation_and_authoritative_resume_are_distinct(self):
        first = json.dumps(formats.message_item('user', 'kept preview'))
        wire = '{"events":[' + first + ',invalid],"session_state":{"cwd":"kept"}}'
        source = MeasuredBytes(wire.encode())
        with self.source(source):
            rows = await savefiles.list_saved_sessions('opaque')
        self.assertEqual(rows[0]['preview'], 'kept preview')
        with self.assertRaises(ValueError):
            savefiles.read_chat_log(io.StringIO(wire))

    async def test_large_metadata_escaped_keys_and_utf8_cross_chunk_boundaries(self):
        cwd = 'raw/../' + '\u96ea\\"{}[]' * 5000 + '/./'
        state = json.dumps({'unwanted': ['}"{\\' * 9000, {'nested': [True, None]}],
                            'cwd': cwd}, ensure_ascii=False)
        state = state.replace('"cwd"', '"\\u0063wd"')
        wire = '{"events":[],"session_\\u0073tate":' + state + '}'
        source = MeasuredBytes(wire.encode('utf-8'))
        with self.source(source):
            rows = await savefiles.list_saved_sessions('opaque')
        self.assertEqual(rows[0]['cwd'], cwd)
        self.assertTrue(source.closed)
        self.assertGreater(source.consumed, json_streams.CHUNK_SIZE)

    async def test_bad_or_unreadable_metadata_is_omitted_not_a_blank_row(self):
        wires = [b'', b'[]', b'{"events":[]}', b'{"session_state":null}',
                 b'{"session_state":{"cwd":""}}', b'{"session_state":{"cwd":3}}',
                 b'{"session_state":{"cwd":"\\q"}}', b'{"session_state":{"cwd":"x"}',
                 b'{"session_state":{"cwd":"x"}} trailing', b'\xff']
        for wire in wires:
            source = MeasuredBytes(wire)
            with self.subTest(wire=wire), self.source(source):
                self.assertEqual(await savefiles.list_saved_sessions('opaque'), [])
            self.assertTrue(source.closed)
        with mock.patch.object(savefiles, 'chat_log_entries', return_value=[SimpleNamespace(name='chat-saved.json', path=object())]), \
                mock.patch.object(savefiles, 'open_chat_log', side_effect=PermissionError('denied')):
            self.assertEqual(await savefiles.list_saved_sessions('opaque'), [])

    async def test_fake_metadata_inside_history_is_not_selected(self):
        fake = json.dumps({'session_state': {'cwd': 'wrong'}})
        source = MeasuredBytes(('{"events":[' + fake + ']}').encode())
        with self.source(source):
            self.assertEqual(await savefiles.list_saved_sessions('opaque'), [])

    async def test_query_is_explicit_full_text_not_just_the_label(self):
        wire = self.wire(256 * 1024)
        for query, expected in [['', 1], ['historical cwd', 1], ['xxxxxxxx raw', 1], ['not-in-the-log', 0]]:
            source = MeasuredBytes(wire)
            with self.subTest(query=query), self.source(source):
                rows = await savefiles.list_saved_sessions('opaque', query=query)
            self.assertEqual(len(rows), expected)
            if query:
                self.assertGreaterEqual(source.consumed, len(wire))
            else:
                self.assertLess(source.consumed, len(wire))

    async def test_directory_and_programming_errors_propagate(self):
        with mock.patch.object(savefiles, 'chat_log_entries', side_effect=PermissionError('directory denied')):
            with self.assertRaises(PermissionError):
                await savefiles.list_saved_sessions('opaque')
        source = MeasuredBytes(self.wire(0))
        with self.source(source), mock.patch.object(source, 'read', side_effect=RuntimeError('bug')):
            with self.assertRaisesRegex(RuntimeError, 'bug'):
                await savefiles.list_saved_sessions('opaque')
        self.assertTrue(source.closed)

    async def test_cancellation_closes_the_same_checked_file_without_more_reads(self):
        source = MeasuredBytes(self.wire(1024 * 1024))
        ready = asyncio.get_running_loop().create_future()
        read = source.read

        def reading(size):
            value = read(size)
            if not ready.done():
                ready.set_result(None)
            return value

        with self.source(source), mock.patch.object(source, 'read', side_effect=reading):
            operation = asyncio.create_task(savefiles.list_saved_sessions('opaque'))
            try:
                await asyncio.wait([ready, operation], return_when=asyncio.FIRST_COMPLETED)
                self.assertFalse(operation.done())
                consumed = source.consumed
                operation.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await operation
                self.assertEqual(source.consumed, consumed)
            finally:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
        self.assertTrue(source.closed)
