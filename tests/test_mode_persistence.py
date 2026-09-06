import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from loki_agent import formats, loki
from loki_agent.sessions import Session


class ModePersistenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = pathlib.Path(directory.name) / "chat.json"
        self.session = Session(shell_cwd=directory.name)
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)
        # Loading saved cwd must not change the test runner's working directory.
        patch = mock.patch.object(loki, "change_shell_cwd")
        patch.start()
        self.addCleanup(patch.stop)
        loki.new_chat_log(str(self.path))

    def save_announcement(self, mode):
        self.session.agent_mode = mode
        loki.record_agent_mode_instruction()
        self.session.transcript_items.append(formats.message_item("user", "Hi"))
        loki.save_chat_log()

    def test_each_mode_round_trips_and_same_mode_is_not_repeated(self):
        for mode in loki.MODE_CYCLE_ORDER:
            with self.subTest(mode=mode):
                loki.new_chat_log(str(self.path))
                self.save_announcement(mode)
                saved = json.loads(self.path.read_text())
                self.assertEqual(
                    saved["session_state"]["last_instructed_agent_mode"], mode)
                self.session.last_instructed_agent_mode = None
                loki.load_chat_log(str(self.path))
                self.assertEqual(self.session.last_instructed_agent_mode, mode)
                before = list(self.session.transcript_items)
                loki.record_agent_mode_instruction()
                self.assertEqual(self.session.transcript_items, before)
                self.assertFalse(self.session.chat_log_dirty)

    def test_resume_does_not_activate_saved_mode_and_change_announces_once(self):
        self.save_announcement("plan")
        self.session.agent_mode = "normal"
        loki.load_chat_log(str(self.path))
        self.assertEqual(self.session.agent_mode, "normal")
        self.assertEqual(self.session.last_instructed_agent_mode, "plan")
        before = len(self.session.transcript_items)
        loki.record_agent_mode_instruction()
        self.assertEqual(len(self.session.transcript_items), before + 1)
        self.assertEqual(self.session.last_instructed_agent_mode, "normal")
        loki.record_agent_mode_instruction()
        self.assertEqual(len(self.session.transcript_items), before + 1)

    def test_missing_or_invalid_marker_is_unknown_without_text_inference(self):
        states = [{}, *({"last_instructed_agent_mode": value}
                        for value in (None, 1, [], {}, "bogus", "Normal"))]
        for state in states:
            with self.subTest(state=state):
                self.save_announcement("normal")
                transcript = list(self.session.transcript_items)
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    loki.load_chat_log(str(self.path), loaded=(
                        transcript, [], state, []))
                self.assertIsNone(self.session.last_instructed_agent_mode)
                self.assertEqual(bool(errors.getvalue()), bool(state))
                before = len(self.session.transcript_items)
                loki.record_agent_mode_instruction()
                self.assertEqual(len(self.session.transcript_items), before + 1)

    def test_pending_turn_without_model_response_is_saved_with_marker(self):
        # A failed/cancelled generation leaves the announcement and user input
        # in the transcript. Persistence must not depend on an acknowledgement.
        self.save_announcement("normal")
        before = list(self.session.transcript_items)
        loki.load_chat_log(str(self.path))
        loki.record_agent_mode_instruction()
        self.assertEqual(self.session.transcript_items, before)

    def test_new_chat_and_loading_another_chat_do_not_leak_marker(self):
        self.save_announcement("plan")
        other = self.path.with_name("other.json")
        loki.new_chat_log(str(other))
        self.assertIsNone(self.session.last_instructed_agent_mode)
        loki.save_chat_log()
        self.assertNotIn("last_instructed_agent_mode",
                         json.loads(other.read_text())["session_state"])
        loki.load_chat_log(str(self.path))
        self.assertEqual(self.session.last_instructed_agent_mode, "plan")
        loki.load_chat_log(str(other))
        self.assertIsNone(self.session.last_instructed_agent_mode)
        loki.record_agent_mode_instruction()
        self.assertEqual(self.session.last_instructed_agent_mode, "plan")

    def test_save_failure_does_not_publish_marker_without_transcript(self):
        self.save_announcement("normal")
        previous = self.path.read_bytes()
        self.session.agent_mode = "plan"
        loki.record_agent_mode_instruction()
        with mock.patch.object(loki, "_atomic_write_text",
                               side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                loki.save_chat_log()
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(self.session.session_state[
            "last_instructed_agent_mode"], "normal")
        self.assertTrue(self.session.chat_log_dirty)
        loki.save_chat_log()
        loki.load_chat_log(str(self.path))
        self.assertEqual(self.session.last_instructed_agent_mode, "plan")

    def test_resume_alone_does_not_append_or_generate(self):
        self.save_announcement("normal")
        before = list(self.session.transcript_items)
        with mock.patch(
                "loki_agent.terminal_frontend.run_terminal_turn_async") as generate:
            loki.load_chat_log(str(self.path))
        generate.assert_not_called()
        self.assertEqual(self.session.transcript_items, before)
        self.assertFalse(self.session.chat_log_dirty)
