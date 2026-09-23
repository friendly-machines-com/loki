import unittest

from loki_agent import subagents
from loki_agent.sessions import Session, conversation_id_for_path


class CodexSessionIdentityTests(unittest.TestCase):
    def test_roots_follow_persistent_conversation_not_runtime_or_saved_fields(self):
        first = Session()
        resumed = Session()
        self.assertNotEqual(first.conversation_id, resumed.conversation_id)
        path = "chat-35b2d314-fdfb-466f-bbd6-4f479fc82eb4.json"
        for session in (first, resumed):
            session.replace_transcript([], [], [], {
                "delegated_root_conversation_id": "untrusted",
                "root_conversation_id": "untrusted",
                "conversation_id": "untrusted",
            }, path)
            self.assertEqual(session.conversation_id, conversation_id_for_path(path))
            self.assertEqual(session.root_conversation_id, session.conversation_id)
        self.assertEqual(first.root_conversation_id, resumed.root_conversation_id)
        resumed.replace_transcript([], [], [], {}, "another-chat.json")
        self.assertNotEqual(first.root_conversation_id, resumed.root_conversation_id)
        self.assertEqual(resumed.root_conversation_id, resumed.conversation_id)

    def test_delegated_tree_identity_does_not_replace_child_thread_identity(self):
        root = Session()
        child = Session(delegated_root_conversation_id=root.root_conversation_id)
        grandchild = Session(
            delegated_root_conversation_id=child.root_conversation_id)
        self.assertEqual(len({
            root.conversation_id, child.conversation_id, grandchild.conversation_id,
        }), 3)
        self.assertEqual(child.root_conversation_id, root.conversation_id)
        self.assertEqual(grandchild.root_conversation_id, root.conversation_id)

    def test_delegated_root_argument_accepts_only_uuid_identity(self):
        identity = "35b2d314-fdfb-466f-bbd6-4f479fc82eb4"
        options = subagents.parse_args([
            "Explore", "--root-conversation-id", identity.upper(),
        ])
        self.assertEqual(options.root_conversation_id, identity)
        for invalid in ("", "not-an-id", identity + "\r\nInjected: yes"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                subagents.parse_args([
                    "Explore", "--root-conversation-id", invalid,
                ])
