"""Command delivery classification: what bypasses the prompt queue, where.

Immediate commands are hypervisor monitor-plane interaction (invariant H, see
the command_deliveries module docstring): their output never becomes
conversation. These tests pin terminal classification and exercise the
separate ACP admission gate, including decision D2 (nothing immediate on
ACP; overlapping requests are rejected, not queued).
"""

import unittest

from loki_agent import command_deliveries


class ClassifyTests(unittest.TestCase):
    def test_ps_is_immediate_in_any_form(self):
        for text in [" /ps ", "/ps", "/ps all", "/ps stop 1", "/ps bad args"]:
            with self.subTest(text=text):
                parsed = command_deliveries.terminal_immediate(text)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.name, "ps")
        self.assertEqual(
            command_deliveries.terminal_immediate("/ps stop 1").argument,
            "stop 1")

    def test_ps_does_not_intercept_paths_or_lookalikes(self):
        for text in ["/ps/file.py", "/ps-extra", "/PS", "/ps\t1", "/psx"]:
            with self.subTest(text=text):
                self.assertIsNone(command_deliveries.terminal_immediate(text))

    def test_status_covers_exactly_the_inspect_and_save_forms(self):
        immediate = [
            "/status", "/status --json", "/status all",
            "/status all --json", "/status --json all", "/status save",
            "/status  all", "  /status",
        ]
        for text in immediate:
            with self.subTest(text=text):
                parsed = command_deliveries.terminal_immediate(text)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.name, "status")
        # Unrecognized arguments keep their old fate (model prompt), and
        # lookalikes never match.
        for text in ["/status foo", "/status --json --json", "/statusx",
                     "/status save extra", "/STATUS"]:
            with self.subTest(text=text):
                self.assertIsNone(command_deliveries.terminal_immediate(text))

    def test_account_is_immediate_only_in_the_read_only_form(self):
        immediate = {
            "/account usage": "usage",
            "/account usage --json": "usage",
            "/account resets": "resets",
        }
        for text, control in immediate.items():
            with self.subTest(text=text):
                parsed = command_deliveries.terminal_immediate(text)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.name, "account")
        # The interactive forms queue: the bare listing, and control+action.
        for text in ["/account", "/account --json", "/account usage reset",
                     "/account usage reset --json", "/accountx"]:
            with self.subTest(text=text):
                self.assertIsNone(command_deliveries.terminal_immediate(text))

    def test_queue_is_immediate_in_any_form(self):
        for text in ["/queue", "/queue texts", "/queue images",
                     "/queue bogus future-subcommand"]:
            with self.subTest(text=text):
                parsed = command_deliveries.terminal_immediate(text)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.name, "queue")
        for text in ["/queuex", "/queue/file", "/QUEUE"]:
            with self.subTest(text=text):
                self.assertIsNone(command_deliveries.terminal_immediate(text))

    def test_undeclared_and_non_command_lines_are_never_immediate(self):
        for text in ["", "prompt text", "!ls", "/model", "/thinking high",
                     "/pwd", "/cd /tmp", "/image /tmp/x.png", "/quit",
                     "/status9", "//status"]:
            with self.subTest(text=text):
                self.assertIsNone(command_deliveries.terminal_immediate(text))
                self.assertIsNone(command_deliveries.classify(text))


class DeliveryDeclarationTests(unittest.TestCase):
    def test_delivery_defaults_to_ordinary_terminal_prompt(self):
        self.assertEqual(command_deliveries.Delivery().terminal,
                         command_deliveries.PROMPT)

    def test_acp_rejects_overlapping_commands_without_changing_the_turn(self):
        # Exercise the actual ACP gate, not unused delivery metadata. The
        # terminal's immediate classification does not alter ACP admission.
        import asyncio
        from loki_agent import acps
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session

        async def scenario():
            session = Session()
            worker = Worker(session, lambda message: None)
            worker.cancel_event.set()
            turn = asyncio.create_task(asyncio.Event().wait())
            worker._prompt_task = turn
            try:
                for text in ["/status", "/status save", "/account usage"]:
                    with self.subTest(text=text), self.assertRaisesRegex(
                            acps.TransportError, "a prompt is already running"):
                        worker._prepare_prompt({
                            "prompt": [{"type": "text", "text": text}]})
                    self.assertFalse(turn.done())
                    self.assertTrue(worker.cancel_event.is_set())
                    self.assertEqual(session.transcript_items, [])
            finally:
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
