"""Command delivery classification: what bypasses the prompt queue, where.

Immediate commands are hypervisor monitor-plane interaction (invariant H, see
the command_deliveries module docstring): their output never becomes
conversation. These tests pin the classification itself -- the single source
both frontends consult -- and the delivery declarations, including decision
D2 (nothing immediate on ACP).
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
    def test_every_declared_command_is_queued_on_acp(self):
        # Decision D2 (command_deliveries docstring): no out-of-band output channel
        # exists on ACP, so nothing is immediate there. Declared, not
        # assumed -- this pins it until that decision is revisited.
        for name, spec in command_deliveries._COMMANDS.items():
            with self.subTest(name=name):
                self.assertEqual(spec.delivery.terminal,
                                 command_deliveries.IMMEDIATE)
                self.assertEqual(spec.delivery.acp, command_deliveries.QUEUED)

    def test_delivery_defaults_to_queued(self):
        delivery = command_deliveries.Delivery()
        self.assertEqual(delivery.terminal, command_deliveries.QUEUED)
        self.assertEqual(delivery.acp, command_deliveries.QUEUED)


if __name__ == "__main__":
    unittest.main()
