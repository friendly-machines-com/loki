"""Historical-user previews must not read/materialize the rest of a chat."""

import io
import json
import unittest
from unittest import mock

from test_json_streams import Fragments
from loki_agent import formats, savefiles


def user(text):
    return formats.message_item('user', text)


class StreamingPreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_historical_user_not_metadata_provider_output_or_compaction_window(self):
        fake = user('not the historical user')
        events = [formats.message_item('system', 'instructions'),
                  {'type': 'model_response', 'items': [fake]},
                  {'type': 'context_replace', 'events': [fake]}, user('actual historical request'),
                  user('later request')]
        for schema in ['day-agent.session.v4', 'day-agent.session.v5']:
            wire = json.dumps({'future': {'nested': fake}, 'schema': schema, 'events': events,
                               'session_state': {'extra': fake}})
            for size in [1, 7, 16 * 1024]:
                with self.subTest(schema=schema, size=size):
                    self.assertEqual(await savefiles.preview(io.StringIO(wire), chunk_size=size), 'actual historical request')

    async def test_arbitrary_field_order_and_json_escaped_keys(self):
        content = [{'text': '  snow \u96ea\n and emoji \U0001f600  ', 'extra': {'role': 'user'}, 'type': 'text'}]
        events = [{'content': content, 'role': 'user', 'type': 'message'},
                  {'role': 'user', 'content': content, 'type': 'message'},
                  {'type': 'message', 'content': content, 'role': 'user'}]
        for event in events:
            wire = json.dumps({'events': [event], 'schema': 'day-agent.session.v5'})
            wire = wire.replace('"events"', '"\\u0065vents"').replace('"role"', '"\\u0072ole"')
            for split in range(len(wire) + 1):
                with self.subTest(order=list(event), split=split):
                    stream = Fragments([wire[:split], wire[split:]])
                    self.assertEqual(await savefiles.preview(stream, chunk_size=31), 'snow \u96ea and emoji \U0001f600')

    async def test_stops_inside_a_huge_first_text_without_reading_its_tail(self):
        prefix = '{"events":[{"type":"message","role":"user","content":[{"type":"text","text":"'
        stream = Fragments([prefix, 'a' * 81], guarded=True)
        self.assertEqual(await savefiles.preview(stream, chunk_size=17), 'a' * 77 + '...')
        self.assertEqual(stream.characters, len(prefix) + 81)

    async def test_short_preview_does_not_need_the_message_or_document_suffix(self):
        prefix = '{"events":[{"type":"message","role":"user","content":[{"type":"text","text":"hello"'
        stream = Fragments([prefix], guarded=True)
        self.assertEqual(await savefiles.preview(stream, chunk_size=13), 'hello')
        self.assertFalse(stream.closed)

    async def test_huge_unknown_values_and_keys_are_skipped_before_the_target(self):
        def pieces():
            yield '{"'
            for _ in range(512):
                yield 'k' * 4096
            yield '":"'
            for _ in range(1024):
                yield 'x' * 4096
            yield '","events":['
            yield json.dumps(user('target after metadata'))
            yield '],"unread":'
            raise AssertionError('unnecessary suffix read')
        self.assertEqual(await savefiles.preview(Fragments(pieces()), chunk_size=4096), 'target after metadata')

    async def test_late_role_cannot_publish_a_preview_before_classification(self):
        for role, expected in [['user', 'a' * 77 + '...'], ['assistant', '']]:
            def pieces():
                yield '{"events":[{"type":"message","content":[{"type":"text","text":"'
                for _ in range(128):
                    yield 'a' * 4096
                yield '"}],"role":' + json.dumps(role) + '}]}'
            with self.subTest(role=role):
                self.assertEqual(await savefiles.preview(Fragments(pieces()), chunk_size=4096), expected)

    async def test_giant_leading_and_trailing_whitespace_do_not_fill_the_capture(self):
        def pieces():
            yield '{"events":[{"type":"message","role":"user","content":[{"type":"text","text":"'
            for _ in range(128):
                yield ' ' * 4096
            yield 'hello'
            for _ in range(128):
                yield '\\n' * 2048
            yield '"'
            raise AssertionError('read beyond complete preview')
        self.assertEqual(await savefiles.preview(Fragments(pieces()), chunk_size=4096), 'hello')

    async def test_limits_and_ellipsis_use_normalized_decoded_characters(self):
        for text in ['a' * 79, 'a' * 80, 'a' * 81, ' a\n b\t c ', '\U0001f600' * 81, '\ud800 literal lone surrogate']:
            normalized = ' '.join(text.split())
            expected = normalized if len(normalized) <= 80 else normalized[:77] + '...'
            for size in [1, 17]:
                with self.subTest(text=repr(text), size=size):
                    wire = json.dumps({'events': [user(text)]})
                    self.assertEqual(await savefiles.preview(io.StringIO(wire), chunk_size=size), expected)

    async def test_first_user_with_no_text_does_not_borrow_a_later_message(self):
        for content in [[], [{'type': 'image', 'source': {'data': 'opaque'}}], [{'type': 'text', 'text': ''}]]:
            events = [formats.message_item('user', content),
                      formats.message_item('assistant', 'not the preview'), user('later user')]
            wire = json.dumps({'events': events})
            self.assertEqual(await savefiles.preview(io.StringIO(wire), chunk_size=5), '')
        wire = json.dumps({'events': [user([{'type': 'image'}, {'type': 'text', 'text': 'first'},
                                            {'type': 'text', 'text': 'second'}])]})
        self.assertEqual(await savefiles.preview(io.StringIO(wire), chunk_size=3), 'first')

    async def test_partial_and_malformed_input_returns_only_a_recognizable_prefix(self):
        start = '{"events":[{"type":"message","role":"user","content":[{"type":"text","text":"'
        for tail, expected in [['hello', 'hello'], ['hello\\uD8', 'hello'], ['hello\\q', 'hello']]:
            for size in [1, 7]:
                with self.subTest(tail=tail, size=size):
                    self.assertEqual(await savefiles.preview(io.StringIO(start + tail), chunk_size=size), expected)
        for wire in ['', '{', '{"events": [', '[]', '{"text":"unrelated truncated',
                     '{"events": [{"type":"message","role":"assistant","content":[{"type":"text","text":"not a user']:
            self.assertEqual(await savefiles.preview(io.StringIO(wire), chunk_size=1), '')

    async def test_unknown_and_non_user_values_can_contain_json_lookalikes(self):
        values = [None, True, 1e-20, float('inf'), {'role': 'user', 'text': 'fake'},
                  '[{"type":"message","role":"user","content":[{"type":"text","text":"fake"}]}]']
        for value in values:
            wire = json.dumps({'metadata': value, 'events': [formats.message_item('developer', value), user('real')]})
            with self.subTest(value=repr(value)):
                self.assertEqual(await savefiles.preview(io.StringIO(wire), chunk_size=2), 'real')

    def test_picker_row_only_formats_an_already_read_summary(self):
        summary = {'id': 'saved', 'path': object(), 'mtime': 0,
                   'preview': 'historical request', 'cwd': 'raw/../cwd/./'}
        with mock.patch.object(savefiles, 'open_chat_log', side_effect=AssertionError('formatter opened a file')):
            row = savefiles.format_picker_row(1, summary)
        self.assertTrue(row.endswith('  saved  historical request'), row)
