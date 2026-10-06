"""The Ask tool's terminal surface: parsing, dialog, and turn wiring.

The ACP leg (worker -> front -> client elicitation) lives in test_acps; here
the same outcome contract is produced by the TUI dialog on the modal input
path shared with /model and /account.
"""

import asyncio
import contextlib
import io
import types
import unittest
from unittest import mock

from loki_agent import formats, loki, protocols, terminal_frontend
from loki_agent.sessions import Session


class RecordingTerminal:
    def __init__(self):
        self.written = []

    def write_text(self, text, multiline=False, file=None):
        self.written.append(text)


class DialogSession:
    """Just the modal surface run_ask_dialog_async touches."""

    def __init__(self, replies, *, interactive=True):
        self.replies = list(replies)
        self.prompts = []
        self.interactive = interactive
        self.reader = types.SimpleNamespace(
            cancel_requested=False, cancel_event=asyncio.Event())

    @contextlib.asynccontextmanager
    async def modal(self):
        yield self

    async def prompt(self, prompt_text='User: ', history=None, *,
                     initial_text=''):
        self.prompts.append(prompt_text)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def options():
    return [("Local", "on disk"), ("Remote", "S3\nin the cloud")]


class ParseAskAnswerTests(unittest.TestCase):
    labels = ["a", "b", "c"]

    def test_a_number_picks_its_label(self):
        self.assertEqual(
            terminal_frontend.parse_ask_answer("2", self.labels),
            {"action": "answered", "answer": "b"})
        self.assertEqual(
            terminal_frontend.parse_ask_answer(" 3 ", self.labels),
            {"action": "answered", "answer": "c"})

    def test_other_text_is_the_user_s_own_answer(self):
        self.assertEqual(
            terminal_frontend.parse_ask_answer("via VPN", self.labels),
            {"action": "answered", "answer": "via VPN"})

    def test_empty_dismisses_and_d_declines(self):
        self.assertEqual(
            terminal_frontend.parse_ask_answer("", self.labels),
            {"action": "cancelled"})
        self.assertEqual(
            terminal_frontend.parse_ask_answer("   ", self.labels),
            {"action": "cancelled"})
        self.assertEqual(
            terminal_frontend.parse_ask_answer("d", self.labels),
            {"action": "declined"})
        self.assertEqual(
            terminal_frontend.parse_ask_answer("D", self.labels),
            {"action": "declined"})

    def test_malformed_choices_re_prompt(self):
        for text in ["0", "4", "-1", "1,1", "1,2"]:
            with self.subTest(text=text):
                self.assertIsNone(
                    terminal_frontend.parse_ask_answer(text, self.labels))

    def test_space_separated_numbers_are_custom_text(self):
        # Only comma-separated numbers count as picks; anything else is the
        # user's own wording, so "1 2" is answered verbatim.
        self.assertEqual(
            terminal_frontend.parse_ask_answer("1 2", self.labels),
            {"action": "answered", "answer": "1 2"})

    def test_multi_select_collects_numbers_in_answer_order(self):
        self.assertEqual(
            terminal_frontend.parse_ask_answer(
                "1,3", self.labels, multi_select=True),
            {"action": "answered", "answer": ["a", "c"]})
        self.assertEqual(
            terminal_frontend.parse_ask_answer(
                "3,1", self.labels, multi_select=True),
            {"action": "answered", "answer": ["c", "a"]})
        self.assertEqual(
            terminal_frontend.parse_ask_answer(
                "2", self.labels, multi_select=True),
            {"action": "answered", "answer": ["b"]})

    def test_multi_select_custom_text_is_a_single_element_answer(self):
        self.assertEqual(
            terminal_frontend.parse_ask_answer(
                "maybe later", self.labels, multi_select=True),
            {"action": "answered", "answer": ["maybe later"]})

    def test_multi_select_repeats_and_outliers_re_prompt(self):
        for text in ["1,1", "0", "9"]:
            with self.subTest(text=text):
                self.assertIsNone(terminal_frontend.parse_ask_answer(
                    text, self.labels, multi_select=True))

    def test_stray_commas_do_not_turn_a_pick_into_text(self):
        self.assertEqual(
            terminal_frontend.parse_ask_answer("1,", self.labels),
            {"action": "answered", "answer": "a"})
        self.assertEqual(
            terminal_frontend.parse_ask_answer(
                "1, 3,", self.labels, multi_select=True),
            {"action": "answered", "answer": ["a", "c"]})
        self.assertEqual(
            terminal_frontend.parse_ask_answer(",1,", self.labels),
            {"action": "answered", "answer": "a"})


class AskDialogTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, session, *, multi_select=False):
        terminal = RecordingTerminal()
        output = io.StringIO()
        with mock.patch.object(terminal_frontend, "terminal", terminal), \
                contextlib.redirect_stdout(output):
            outcome = await terminal_frontend.run_ask_dialog_async(
                session, session.reader.cancel_event, "Which storage?",
                options(), multi_select=multi_select)
        return outcome, terminal, output

    async def test_dialog_renders_question_options_and_descriptions(self):
        session = DialogSession(["2"])
        outcome, terminal, _ = await self._run(session)
        self.assertEqual(
            outcome, {"action": "answered", "answer": "Remote"})
        self.assertEqual(terminal.written[0], "Which storage?")
        self.assertIn("1. Local", terminal.written)
        self.assertIn("2. Remote", terminal.written)
        self.assertIn("   on disk", terminal.written)
        self.assertIn("   S3\n   in the cloud", terminal.written)

    async def test_multi_select_dialog_returns_the_picked_labels(self):
        session = DialogSession(["1,2"])
        outcome, _, _ = await self._run(session, multi_select=True)
        self.assertEqual(
            outcome, {"action": "answered", "answer": ["Local", "Remote"]})

    async def test_custom_text_decline_and_dismiss(self):
        for reply, expected in [
                ("via VPN", {"action": "answered", "answer": "via VPN"}),
                ("d", {"action": "declined"}),
                ("", {"action": "cancelled"})]:
            with self.subTest(reply=reply):
                outcome, _, _ = await self._run(DialogSession([reply]))
                self.assertEqual(outcome, expected)

    async def test_bad_choice_re_prompts_then_answers(self):
        session = DialogSession(["9", "1"])
        outcome, _, _ = await self._run(session)
        self.assertEqual(outcome, {"action": "answered", "answer": "Local"})
        # The invalid choice re-prompted before the valid one was accepted.
        self.assertEqual(len(session.prompts), 2)

    async def test_cancel_event_settles_the_open_question(self):
        session = DialogSession(["1"])
        session.reader.cancel_event.set()
        outcome, _, _ = await self._run(session)
        self.assertEqual(outcome, {"action": "cancelled"})

    async def test_ctrl_c_and_eof_dismiss_without_hanging_the_turn(self):
        for error in (KeyboardInterrupt(), EOFError()):
            with self.subTest(error=type(error).__name__):
                outcome, _, _ = await self._run(DialogSession([error]))
                self.assertEqual(outcome, {"action": "cancelled"})


class TerminalAskUserTests(unittest.IsolatedAsyncioTestCase):
    def test_noninteractive_sessions_get_no_seam(self):
        session = DialogSession([], interactive=False)
        self.assertIsNone(terminal_frontend.terminal_ask_user(session))

    async def test_the_seam_routes_through_the_dialog(self):
        session = DialogSession(["2"])
        ask_user = terminal_frontend.terminal_ask_user(session)
        with mock.patch.object(terminal_frontend, "terminal",
                               RecordingTerminal()), \
                contextlib.redirect_stdout(io.StringIO()):
            outcome = await ask_user(
                "Which storage?", [("Local", "on disk"), ("Remote", "S3")])
        self.assertEqual(outcome, {"action": "answered", "answer": "Remote"})


class PlanModeAdvertisementTests(unittest.TestCase):
    def test_ask_is_a_plan_tool_and_never_an_explore_tool(self):
        self.assertIn("Ask", loki.PLAN_TOOLS)
        self.assertNotIn("Ask", loki.EXPLORE_TOOLS)


class TurnAdvertisementTests(unittest.IsolatedAsyncioTestCase):
    """The turn advertises Ask exactly when it carries a working seam."""

    def setUp(self):
        self.session = Session(runtime_config=loki.make_runtime_config(
            "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES,
            model="claude-opus-4-5"))
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    async def _turn(self, **kwargs):
        advertised = []

        async def completion(items, tools=None, *args, **kwargs):
            advertised.append(
                [tool["function"]["name"] for tool in tools])
            return formats.DecodedTurn(
                [formats.message_item("assistant", "answer")])

        with mock.patch.object(
                    terminal_frontend, "async_chat_completion", completion), \
                mock.patch.object(terminal_frontend, "_terminal_agent_event"), \
                mock.patch.object(terminal_frontend.terminals,
                                  "redraw_status_bar"), \
                contextlib.redirect_stdout(io.StringIO()):
            await terminal_frontend.run_terminal_turn_async([], **kwargs)
        return advertised[0]

    async def test_no_seam_hides_ask(self):
        self.assertNotIn("Ask", await self._turn())

    async def test_a_seam_advertises_ask(self):
        async def ask_user(question, options, *, multi_select=False):
            return {"action": "cancelled"}

        self.assertIn("Ask", await self._turn(ask_user=ask_user))


class LoopAdvertisementTests(unittest.IsolatedAsyncioTestCase):
    """run_tool_loop_async strips Ask for seam-less callers (subagents)."""

    def setUp(self):
        self.session = Session(runtime_config=loki.make_runtime_config(
            "https://api.anthropic.com/v1", protocols.ANTHROPIC_MESSAGES,
            model="claude-opus-4-5"))
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    async def _loop(self, **kwargs):
        advertised = []

        async def completion(items, *args, **kwargs):
            advertised.append(
                [tool["function"]["name"] for tool in kwargs["tools"]])
            return formats.DecodedTurn(
                [formats.message_item("assistant", "answer")])

        with mock.patch.object(loki, "async_chat_completion", completion), \
                contextlib.redirect_stdout(io.StringIO()):
            await loki.run_tool_loop_async([], **kwargs)
        return advertised[0]

    async def test_no_seam_hides_ask(self):
        self.assertNotIn("Ask", await self._loop())

    async def test_a_seam_advertises_ask(self):
        async def ask_user(question, options, *, multi_select=False):
            return {"action": "cancelled"}

        self.assertIn("Ask", await self._loop(ask_user=ask_user))


class AskHeaderTests(unittest.TestCase):
    """The tool's public contract is stated once and shared by every owner."""

    def _ask_definition(self):
        for tool in loki.TOOLS:
            if tool["function"]["name"] == loki.ASK_TOOL_NAME:
                return tool
        self.fail("Ask tool missing from the registry")

    def test_name_constant_and_registry_agree(self):
        self.assertIsInstance(loki.ASK_TOOL_NAME, str)
        self.assertIn(loki.ASK_TOOL_NAME, loki.TOOL_REGISTRY)
        self.assertIn(loki.ASK_TOOL_NAME, loki.PLAN_TOOLS)
        self.assertNotIn(loki.ASK_TOOL_NAME, loki.EXPLORE_TOOLS)

    def test_schema_bounds_come_from_the_shared_constants(self):
        parameters = self._ask_definition()["function"]["parameters"]
        properties = parameters["properties"]
        options = properties["options"]
        self.assertEqual(options["minItems"], loki.ASK_MIN_OPTIONS)
        self.assertEqual(options["maxItems"], loki.ASK_MAX_OPTIONS)
        self.assertEqual(
            properties["question"]["maxLength"], loki.ASK_MAX_QUESTION_CHARS)
        item = options["items"]["properties"]
        self.assertEqual(item["label"]["maxLength"], loki.ASK_MAX_LABEL_CHARS)
        self.assertEqual(
            item["description"]["maxLength"], loki.ASK_MAX_DESCRIPTION_CHARS)

    def test_without_ask_tool_removes_only_the_ask_tool(self):
        rest = loki.without_ask_tool(loki.TOOLS)
        names = [tool["function"]["name"] for tool in rest]
        self.assertNotIn(loki.ASK_TOOL_NAME, names)
        self.assertEqual(len(rest), len(loki.TOOLS) - 1)


class AskHandlerTests(unittest.IsolatedAsyncioTestCase):
    """The outcome-to-text renderer every Ask path funnels through."""

    async def _run(self, args, outcome):
        self.last = {}

        async def ask_user(question, options, *, multi_select=False):
            self.last = {
                "question": question, "options": options,
                "multi_select": multi_select}
            return outcome

        text = await loki._handle_ask_async(args, {"ask_user": ask_user})
        return text

    def _args(self, **over):
        args = {
            "question": "Which storage?",
            "options": [
                {"label": "Local", "description": "on disk"},
                {"label": "Remote"},
            ],
        }
        args.update(over)
        return args

    async def test_no_seam_answers_without_invoking_the_user(self):
        self.assertIn("not available", await loki._handle_ask_async(
            self._args(), None))

    async def test_malformed_inputs_are_rejected_before_asking(self):
        cases = [
            ({}, "at least two options"),
            ({"question": "q?", "options": []}, "at least two options"),
            ({"question": "q?", "options": [{"label": "a"}]},
             "at least two options"),
            ({"question": "q?", "options": [{"label": "a"}, "Remote"]},
             "objects with a label"),
            ({"question": "q?", "options": [{"label": "a"}, {"label": 5}]},
             "nonempty strings"),
            ({"question": "q?", "options": [{"label": "a"}, {"label": "a"}]},
             "unique"),
            ({"question": "  ", "options": [{"label": "a"}, {"label": "b"}]},
             "nonempty question"),
        ]
        for args, pattern in cases:
            with self.subTest(args=args):
                called = []

                async def ask_user(question, options, *, multi_select=False):
                    called.append(question)

                text = await loki._handle_ask_async(
                    args, {"ask_user": ask_user})
                self.assertIn(pattern, text)
                self.assertEqual(called, [])

    async def test_answer_is_rendered_with_its_question_and_note(self):
        text = await self._run(
            self._args(),
            {"action": "answered", "answer": "Remote", "custom": "via VPN"})
        self.assertIn('"Which storage?"', text)
        self.assertIn('"Remote"', text)
        self.assertIn("via VPN", text)
        self.assertEqual(self.last["options"], [
            ("Local", "on disk"), ("Remote", None)])
        self.assertFalse(self.last["multi_select"])

    async def test_decline_and_dismiss_are_distinct(self):
        self.assertIn("declined", await self._run(
            self._args(), {"action": "declined"}))
        self.assertIn("dismissed", await self._run(
            self._args(), {"action": "cancelled"}))

    async def test_multi_select_answer_is_rendered_as_a_quoted_list(self):
        text = await self._run(
            self._args(), {"action": "answered", "answer": ["Legs", "Roof"]})
        self.assertIn('"Legs"', text)
        self.assertIn('"Roof"', text)

    async def test_a_non_dict_outcome_is_a_delivery_failure(self):
        self.assertIn("could not be delivered", await self._run(
            self._args(), None))


if __name__ == "__main__":
    unittest.main()
