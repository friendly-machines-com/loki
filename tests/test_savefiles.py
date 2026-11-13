"""Saved-conversation identifier, lookup, and picker validation."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock


class SavedConversationNameTests(unittest.TestCase):
    def test_saved_identifiers_obey_the_same_allowlist_as_new_names(self):
        """Existing or renamed files cannot enlarge the identifier alphabet."""
        from loki_agent import savefiles

        with tempfile.TemporaryDirectory() as root:
            named = os.path.join(root, 'chat-invalid name.json')
            plain = os.path.join(root, 'chat-saved.json')
            wrapped = os.path.join(root, 'chat-chat-saved.json.json')
            for path in [named, plain, wrapped]:
                with open(path, 'w', encoding='utf-8') as stream:
                    stream.write('{}')
            self.assertEqual([entry.name for entry in savefiles.chat_log_entries(root)], ['chat-saved.json'])
            self.assertEqual(savefiles.saved_chat_log_path(root, 'saved'), plain)
            for value in ['', '../outside', 'nested/name', 'invalid\x00name', 'invalid name',
                          'bad\u2603', 'bad\x1b[31m', 'bad\udc9b31m', named,
                          'chat-saved.json', 'chat-chat-saved.json.json']:
                with self.subTest(value=value):
                    self.assertFalse(savefiles.valid_chat_id(value))
                    self.assertIsNone(savefiles.saved_chat_log_path(root, value))
                    with self.assertRaises(ValueError):
                        savefiles.new_chat_log_path(root, value)
            # The storage component is chat-NUL.json, not the DOS device NUL.
            self.assertEqual(savefiles.new_chat_log_path(root, 'NUL'),
                             os.path.join(root, 'chat-NUL.json'))
            self.assertEqual(len(os.listdir(root)), 3)
            os.replace(plain, named)
            self.assertEqual(savefiles.chat_log_entries(root), [])
            self.assertIsNone(savefiles.saved_chat_log_path(root, 'saved'))
            self.assertIsNone(savefiles.saved_chat_log_path(root, 'invalid name'))

    def test_hostile_directory_names_are_filtered_before_lookup_or_display(self):
        from types import SimpleNamespace
        from loki_agent import savefiles

        allowed = SimpleNamespace(name='chat-safe_123-ID.json', path=mock.sentinel.allowed_path)
        invalid = [SimpleNamespace(name='chat-' + value + '.json', path=mock.sentinel.hostile_path)
                   for value in ['', 'bad name', 'bad\u2603', '\x1b[31m', '\x9b31m', '\udc9b31m',
                                 'bad\nname', 'chat-safe.json']]
        with mock.patch.object(savefiles.os, 'scandir') as scan, \
                mock.patch.object(savefiles, 'open_chat_log') as read, \
                mock.patch.object(savefiles.os, 'stat') as metadata:
            scan.return_value.__enter__.return_value = [allowed, *invalid]
            self.assertEqual(savefiles.chat_log_entries('unused'), [allowed])
            self.assertIs(savefiles.saved_chat_log_path('unused', 'safe_123-ID'), allowed.path)
            for entry in invalid:
                with self.subTest(name=entry.name):
                    self.assertIsNone(savefiles.saved_chat_log_path('unused', entry.name[5:-5]))
            read.assert_not_called()
            read.side_effect = OSError('unreadable fixture')
            self.assertEqual(asyncio.run(savefiles.list_saved_sessions('unused')), [])
            read.assert_called_once_with(allowed.path)
            metadata.assert_not_called()

    def test_bare_cli_saved_names_cannot_bypass_the_allowlist(self):
        from loki_agent import savefiles

        resolver = mock.Mock()
        for value in ['bad name', 'bad\u2603', '\x1b[31m', '\x9b31m', '\udc9b31m',
                      'chat-.json', 'chat-bad\u2603.json']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                savefiles.resolve_chat_log_path(value, 'unused', 'unused', resolver)
        resolver.assert_not_called()
