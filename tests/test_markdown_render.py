"""Bounded incremental Markdown rendering and terminal-event wiring."""

import asyncio
import contextlib
import io
import random
import types
import unittest
from unittest import mock


from loki_agent import formats
from loki_agent import loki
from loki_agent import terminal_frontend
from loki_agent import terminals


BOLD = "\033[1m"
ITALIC = "\033[3m"
CODE = "\033[36m"
RESET = "\033[0m"
HEADLINE = "\033[42m"
HEADLINE_OFF = "\033[49m"
BOLD_OFF = "\033[22m"
ITALIC_OFF = "\033[23m"
FOREGROUND_OFF = "\033[39m"


class PresentationTerminal(terminals._TerminalTextOutput):
    def __init__(self, *, style):
        self.markdown_style = style
        self.assistant_markdown = terminals.AssistantMarkdownPresentation(
            self)

    def set_background_color(self, _index):
        pass

    def set_foreground_color(self, _index):
        pass

    def reset_colors_and_flags(self):
        pass


class StyledTerminal(PresentationTerminal):
    def __init__(self):
        super().__init__(style=True)


class NoneTerminal(PresentationTerminal):
    def __init__(self):
        super().__init__(style=False)


def split_at(text, cuts):
    pieces = []
    last = 0
    for cut in sorted(set(c for c in cuts if 0 < c < len(text))):
        pieces.append(text[last:cut])
        last = cut
    pieces.append(text[last:])
    return pieces


DOCUMENT = (
    "Plain text arrives now; **bold** and *emphasis* wait only for their "
    "closers, then `code` is styled.\n"
    "Unclosed **bold stays literal across the newline\n"
    "````python\n"
    "x = a * b ** c  # never styled inside a fence\n"
    "``` does not close a four-backtick fence\n"
    "````\n"
    "~~~text\n"
    "*tilde-fenced content is raw*\n"
    "~~~\n"
    "# Title with *emphasis* and `code` inside\n"
    "##NoSpace stays literal\n"
    "### Deep **bold** headline\n"
    "####### seven hashes stay literal\n"
    "After fences, **styling resumes**."
)


class BoundedRendererTests(unittest.TestCase):
    def renderer(self, **kwargs):
        return terminals.BoundedMarkdownAnsi(style=True, **kwargs)

    @staticmethod
    def collect(renderer, chunks):
        output = []
        for chunk in chunks:
            output.extend(renderer.feed(chunk))
        output.extend(renderer.finish())
        return "".join(output)

    def test_decided_plain_prefix_is_emitted_before_newline_or_finish(self):
        renderer = self.renderer()

        self.assertEqual(renderer.feed("ordinary text"), ("ordinary text",))
        self.assertEqual(renderer.retained_characters, 0)

    def test_only_unresolved_inline_suffix_is_retained(self):
        renderer = self.renderer()

        self.assertEqual(renderer.feed("prefix **bo"), ("prefix ",))
        self.assertEqual(renderer.retained_characters, len("**bo"))
        self.assertEqual(
            renderer.feed("ld** suffix"),
            (f"{BOLD}bold{RESET} suffix",),
        )
        self.assertEqual(renderer.retained_characters, 0)

    def test_completed_inline_constructs_are_terminal_neutral(self):
        renderer = self.renderer()
        fragments = renderer.feed("`code` *em* **bold**")

        self.assertEqual(
            "".join(fragments),
            f"{CODE}code{RESET} {ITALIC}em{RESET} "
            f"{BOLD}bold{RESET}",
        )
        for fragment in fragments:
            opens = (
                fragment.count(BOLD)
                + fragment.count(ITALIC)
                + fragment.count(CODE)
            )
            self.assertEqual(fragment.count(RESET), opens)

    def test_source_controls_are_neutralized_before_markdown_styling(self):
        logical = (
            "before \x1b]0;owned\x07 **bold** "
            "\u009b31m \U0001f469\u200d\U0001f4bb")

        rendered = self.collect(
            self.renderer(),
            ["before \x1b]", "0;owned\x07 **bo",
             "ld** \u009b31m \U0001f469\u200d\U0001f4bb"],
        )

        self.assertEqual(
            rendered,
            "before ^[]0;owned^G "
            f"{BOLD}bold{RESET} "
            "\\x9b31m \U0001f469\u200d\U0001f4bb",
        )
        self.assertNotIn("\x1b]0;owned\x07", rendered)
        self.assertEqual(
            rendered,
            terminals.render_markdown(logical, style=True),
        )

    def test_non_ansi_mode_still_neutralizes_source_controls(self):
        renderer = terminals.BoundedMarkdownAnsi(style=False)

        self.assertEqual(
            self.collect(renderer, ["a\x1b", "[2J\nb\u009b"]),
            "a^[[2J\nb\\x9b",
        )

    def test_unclosed_span_becomes_literal_at_newline(self):
        renderer = self.renderer()

        self.assertEqual(
            renderer.feed("before **unfinished\nnext"),
            ("before **unfinished\nnext",),
        )
        self.assertEqual(renderer.retained_characters, 0)

    def test_unclosed_span_becomes_literal_at_finish(self):
        renderer = self.renderer()

        self.assertEqual(renderer.feed("before `unfinished"), ("before ",))
        self.assertEqual(renderer.finish(), ("`unfinished",))
        self.assertEqual(renderer.finish(), ())

    def test_unresolved_span_has_a_hard_bound_and_overflows_literal(self):
        renderer = self.renderer(max_unresolved=8)
        text = "before `12345678 after"
        output = []

        for character in text:
            output.extend(renderer.feed(character))
            self.assertLessEqual(renderer.retained_characters, 8)
        output.extend(renderer.finish())

        self.assertEqual("".join(output), text)

    def test_overflow_rule_is_the_same_in_batch_and_streaming(self):
        text = "prefix **123456789 suffix *ok*"
        expected = terminals.render_markdown(
            text, style=True, max_unresolved=8)

        for chunks in [
                [text],
                list(text),
                split_at(text, [2, 9, 14, 18, 24])]:
            with self.subTest(chunks=chunks):
                renderer = self.renderer(max_unresolved=8)
                self.assertEqual(self.collect(renderer, chunks), expected)

    def test_overflow_keeps_the_rest_of_its_line_literal(self):
        text = "**123456789** between **ok**\n*next*"

        self.assertEqual(
            terminals.render_markdown(
                text, style=True, max_unresolved=8),
            text.split("\n", 1)[0] + "\n" + f"{ITALIC}next{RESET}",
        )

    def test_fenced_content_streams_raw_once_opening_is_known(self):
        renderer = self.renderer()

        self.assertEqual(renderer.feed("```py"), ("```py",))
        self.assertEqual(
            renderer.feed("\nx = *not emphasis*"),
            ("\nx = *not emphasis*",),
        )
        self.assertEqual(renderer.retained_characters, 0)
        self.assertEqual(renderer.feed("\n```"), ("\n",))
        self.assertEqual(renderer.finish(), ("```",))

    def test_closing_fence_requires_matching_marker_and_length(self):
        renderer = self.renderer()
        text = (
            "````\n"
            "raw *one*\n"
            "```\n"
            "raw **two**\n"
            "```` trailing text\n"
            "raw `three`\n"
            "````\n"
            "**styled**"
        )

        self.assertEqual(
            self.collect(renderer, list(text)),
            (
                "````\n"
                "raw *one*\n"
                "```\n"
                "raw **two**\n"
                "```` trailing text\n"
                "raw `three`\n"
                "````\n"
                f"{BOLD}styled{RESET}"
            ),
        )

    def test_tilde_fences_are_verbatim(self):
        text = "~~~python\n*raw*\n~~~\n*styled*"

        self.assertEqual(
            terminals.render_markdown(text, style=True),
            f"~~~python\n*raw*\n~~~\n{ITALIC}styled{RESET}",
        )

    def test_non_ansi_mode_is_lossless(self):
        renderer = terminals.BoundedMarkdownAnsi(style=False)

        self.assertEqual(
            self.collect(renderer, split_at(DOCUMENT, [3, 17, 49, 100])),
            DOCUMENT,
        )

    def test_non_ansi_mode_never_buffers_markdown(self):
        renderer = terminals.BoundedMarkdownAnsi(style=False)

        self.assertEqual(renderer.feed("**unfinished"), ("**unfinished",))
        self.assertEqual(renderer.retained_characters, 0)
        self.assertEqual(renderer.finish(), ())

    def test_leading_fence_candidate_is_the_only_plain_prefix_delay(self):
        renderer = self.renderer()

        self.assertEqual(renderer.feed("   "), ())
        self.assertEqual(renderer.retained_characters, 3)
        self.assertEqual(renderer.feed("ordinary"), ("   ordinary",))
        self.assertEqual(renderer.retained_characters, 0)

    def test_feed_after_finish_is_rejected(self):
        renderer = self.renderer()
        renderer.finish()

        with self.assertRaisesRegex(RuntimeError, "finished"):
            renderer.feed("late")

    def test_invalid_bound_is_rejected(self):
        for value in [0, -1, True, 1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.renderer(max_unresolved=value)

    def test_headline(self):
        renderer = self.renderer()

        self.assertEqual(renderer.feed("##"), ())
        self.assertEqual(renderer.retained_characters, 2)
        self.assertEqual(renderer.feed(" Foo"), ())
        self.assertEqual(renderer.retained_characters, 6)
        self.assertEqual(
            renderer.feed("\n"),
            (f"{HEADLINE}## Foo{HEADLINE_OFF}\n",))
        self.assertEqual(renderer.retained_characters, 0)
        self.assertEqual(renderer.finish(), ())

    def test_faux_headline_1(self):
        renderer = self.renderer()

        self.assertEqual(renderer.feed("##"), ())
        self.assertEqual(renderer.retained_characters, 2)
        self.assertEqual(renderer.feed("Foo"), ("##Foo",))
        self.assertEqual(renderer.retained_characters, 0)
        self.assertEqual(renderer.feed("\n"), ("\n",))
        self.assertEqual(renderer.retained_characters, 0)
        self.assertEqual(renderer.finish(), ())

    def test_headline_styles_inner_spans(self):
        self.assertEqual(
            terminals.render_markdown(
                "# Hello *foo* and `bar` and **baz**\n", style=True),
            f"{HEADLINE}# Hello {ITALIC}foo{ITALIC_OFF} and "
            f"{CODE}bar{FOREGROUND_OFF} and {BOLD}baz{BOLD_OFF}"
            f"{HEADLINE_OFF}\n",
        )

    def test_faux_headline_resumes_inline_scanning(self):
        self.assertEqual(
            terminals.render_markdown("##**b** after\n", style=True),
            f"##{BOLD}b{RESET} after\n",
        )

    def test_headline_marker_alone_is_literal(self):
        self.assertEqual(
            terminals.render_markdown("#\n##\n", style=True), "#\n##\n")

    def test_unclosed_headline_is_literal(self):
        renderer = self.renderer()

        self.assertEqual(renderer.feed("## never closed"), ())
        self.assertEqual(renderer.retained_characters, 15)
        self.assertEqual(renderer.finish(), ("## never closed",))
        self.assertEqual(renderer.retained_characters, 0)

    def test_headline_overflow_goes_literal(self):
        renderer = self.renderer(max_unresolved=8)

        self.assertEqual(renderer.feed("## "), ())
        self.assertEqual(renderer.retained_characters, 3)
        # The ninth pending character flushes the span literally and the
        # rest of the line stays literal.
        self.assertEqual(renderer.feed("0123456789"), ("## 0123456789",))
        self.assertEqual(renderer.retained_characters, 0)
        # Overflow literalizes only that line; the next line scans afresh.
        self.assertEqual(
            renderer.feed("\nmore `code`\n"),
            (f"\nmore {CODE}code{RESET}\n",))


class DifferentialTests(unittest.TestCase):
    """Chunk divisions cannot change the final rendering."""

    @staticmethod
    def _ansi_channels_closed(fragment):
        """Every SGR channel opened in the fragment must close inside it.

        Headlines close inner spans with parameter-specific resets while the
        background stays on, so counting RESET alone no longer proves the
        invariant; simulate the channel state instead.
        """
        opens = [
            (BOLD, "bold"), (ITALIC, "italic"), (CODE, "code"),
            (HEADLINE, "bg"),
        ]
        closes = [
            (BOLD_OFF, "bold"), (ITALIC_OFF, "italic"),
            (FOREGROUND_OFF, "code"), (HEADLINE_OFF, "bg"),
        ]
        state = {}
        i = 0
        while i < len(fragment):
            if fragment.startswith(RESET, i):
                state.clear()
                i += len(RESET)
                continue
            for escape, channel in opens:
                if fragment.startswith(escape, i):
                    state[channel] = state.get(channel, 0) + 1
                    i += len(escape)
                    break
            else:
                for escape, channel in closes:
                    if fragment.startswith(escape, i):
                        state[channel] = state.get(channel, 0) - 1
                        if state[channel] < 0:
                            return False
                        i += len(escape)
                        break
                else:
                    i += 1
        return all(count == 0 for count in state.values())

    @classmethod
    def stream_concat(cls, text, cuts, *, max_unresolved=4096):
        renderer = terminals.BoundedMarkdownAnsi(
            style=True, max_unresolved=max_unresolved)
        output = []
        for chunk in split_at(text, cuts):
            fragments = renderer.feed(chunk)
            for fragment in fragments:
                if not cls._ansi_channels_closed(fragment):
                    raise AssertionError(
                        f"ANSI state escaped a fragment: {fragment!r}")
            output.extend(fragments)
        output.extend(renderer.finish())
        return "".join(output)

    def assert_chunk_invariant(self, text, cuts, *, max_unresolved=4096):
        expected = terminals.render_markdown(
            text, style=True, max_unresolved=max_unresolved)
        self.assertEqual(
            self.stream_concat(
                text, cuts, max_unresolved=max_unresolved),
            expected,
        )

    def test_every_single_split(self):
        for cut in range(len(DOCUMENT) + 1):
            with self.subTest(cut=cut):
                self.assert_chunk_invariant(DOCUMENT, [cut])

    def test_char_by_char(self):
        self.assert_chunk_invariant(
            DOCUMENT, list(range(1, len(DOCUMENT))))

    def test_random_splits(self):
        rng = random.Random(20260815)
        for _ in range(200):
            cuts = [
                rng.randrange(1, len(DOCUMENT))
                for _ in range(rng.randint(1, 20))
            ]
            with self.subTest(cuts=cuts):
                self.assert_chunk_invariant(DOCUMENT, cuts)

    def test_random_splits_with_small_overflow_bound(self):
        text = "plain **" + ("x" * 40) + "** after `short`"
        rng = random.Random(815)
        for _ in range(100):
            cuts = [
                rng.randrange(1, len(text))
                for _ in range(rng.randint(1, 10))
            ]
            with self.subTest(cuts=cuts):
                self.assert_chunk_invariant(
                    text, cuts, max_unresolved=12)


class TerminalWiringTests(unittest.TestCase):
    def setUp(self):
        self._old_config = loki.current_session().runtime_config
        loki.current_session().runtime_config = types.SimpleNamespace(model="local-model")
        terminal_frontend.terminal.assistant_markdown.reset()

    def tearDown(self):
        terminal_frontend.terminal.assistant_markdown.reset()
        loki.current_session().runtime_config = self._old_config

    def replay(self, events, terminal_stub):
        output = io.StringIO()
        with mock.patch.object(
                terminal_frontend, "terminal", terminal_stub), \
                contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(io.StringIO()):
            for event in events:
                terminal_frontend._terminal_agent_event(event)
        return output.getvalue()

    @staticmethod
    def stream_events(chunks):
        events = [{"type": "assistant_start"}]
        events.extend(
            {"type": "assistant_delta", "content": chunk}
            for chunk in chunks
        )
        events.append({"type": "assistant_end", "complete": True})
        return events

    def test_plain_delta_is_visible_before_assistant_end(self):
        output = io.StringIO()
        with mock.patch.object(
                terminal_frontend, "terminal", StyledTerminal()), \
                contextlib.redirect_stdout(output):
            terminal_frontend._terminal_agent_event({"type": "assistant_start"})
            terminal_frontend._terminal_agent_event({
                "type": "assistant_delta",
                "content": "visible without a newline",
            })
            before_end = output.getvalue()
            terminal_frontend._terminal_agent_event({
                "type": "assistant_end",
                "complete": True,
            })

        self.assertEqual(
            before_end,
            "\nlocal-model: visible without a newline",
        )
        self.assertEqual(output.getvalue(), before_end + "\n")

    def test_streamed_turn_renders_like_batch(self):
        full = "prefix **hello** and `world` done"
        chunks = ["prefix **hel", "lo** and `wor", "ld` do", "ne"]
        streamed = self.replay(
            self.stream_events(chunks), StyledTerminal())
        batch = self.replay(
            [{"type": "assistant_message", "content": full}],
            StyledTerminal())

        self.assertEqual(
            streamed,
            f"\nlocal-model: prefix {BOLD}hello{RESET} and "
            f"{CODE}world{RESET} done\n",
        )
        self.assertEqual(streamed, batch)

    def test_batch_and_split_stream_neutralize_model_terminal_controls(self):
        full = "before \x1b]0;owned\x07 **bold** after\u009b"
        chunks = ["before \x1b]", "0;owned", "\x07 **bo", "ld** after\u009b"]

        streamed = self.replay(
            self.stream_events(chunks), StyledTerminal())
        batch = self.replay(
            [{"type": "assistant_message", "content": full}],
            StyledTerminal())

        self.assertEqual(streamed, batch)
        self.assertNotIn("\x1b]0;owned\x07", streamed)
        self.assertIn("^[]0;owned^G", streamed)
        self.assertIn("\\x9b", streamed)
        self.assertIn(f"{BOLD}bold{RESET}", streamed)

    def test_multiline_tool_error_is_readable_but_controls_are_not_raw(self):
        output = self.replay(
            [{
                "type": "tool_error",
                "result": "first\x1b]0;owned\x07\nsecond\u009b",
            }],
            StyledTerminal(),
        )

        self.assertEqual(
            output, "first^[]0;owned^G\nsecond\\x9b\n")

    def test_tool_name_and_arguments_are_programming_representations(self):
        output = self.replay(
            [{
                "type": "tool_call",
                "name": "Read\x1b]0;owned\x07\nnext",
                "args": {
                    "path": "file\x1b]0;path-owned\x07\nname",
                    "label": "\u6a21\u578b",
                },
            }],
            StyledTerminal(),
        )

        self.assertNotIn("\x1b]0;owned\x07", output)
        self.assertNotIn("\x1b]0;path-owned\x07", output)
        self.assertIn("\\x1b", output)
        self.assertIn("\\nnext", output)
        self.assertIn("\u6a21\u578b", output)

    def test_non_tty_terminal_emits_original_markdown(self):
        streamed = self.replay(
            self.stream_events(["**hel", "lo**"]), NoneTerminal())
        batch = self.replay(
            [{"type": "assistant_message", "content": "**hello**"}],
            NoneTerminal())

        self.assertEqual(streamed, "\nlocal-model: **hello**\n")
        self.assertEqual(streamed, batch)

    def test_delta_without_start_is_finished_safely(self):
        output = self.replay(
            [
                {"type": "assistant_delta", "content": "**a**"},
                {"type": "assistant_end", "complete": True},
            ],
            StyledTerminal(),
        )

        self.assertEqual(output, f"{BOLD}a{RESET}\n")

    def test_second_start_flushes_old_literal_tail(self):
        events = [
            {"type": "assistant_start"},
            {"type": "assistant_delta", "content": "first **unfinished"},
            {"type": "assistant_start"},
            {"type": "assistant_delta", "content": "second"},
            {"type": "assistant_end", "complete": True},
        ]

        self.assertEqual(
            self.replay(events, StyledTerminal()),
            (
                "\nlocal-model: first **unfinished"
                "\nlocal-model: second\n"
            ),
        )

    def test_incomplete_end_flushes_literal_and_clears_state(self):
        events = [
            {"type": "assistant_start"},
            {"type": "assistant_delta", "content": "partial **unfinished"},
            {"type": "assistant_end", "complete": False, "reason": "error"},
            {"type": "assistant_start"},
            {"type": "assistant_delta", "content": "**new**"},
            {"type": "assistant_end", "complete": True},
        ]

        self.assertEqual(
            self.replay(events, StyledTerminal()),
            (
                "\nlocal-model: partial **unfinished\n"
                f"\nlocal-model: {BOLD}new{RESET}\n"
            ),
        )

    def test_end_resets_state_for_next_turn(self):
        events = self.stream_events(["turn **one**"]) + self.stream_events(
            ["turn **two**"])

        self.assertEqual(
            self.replay(events, StyledTerminal()),
            (
                f"\nlocal-model: turn {BOLD}one{RESET}\n"
                f"\nlocal-model: turn {BOLD}two{RESET}\n"
            ),
        )


class ToolLoopWiringTests(unittest.TestCase):
    def setUp(self):
        self._old_config = loki.current_session().runtime_config
        loki.current_session().runtime_config = types.SimpleNamespace(model="local-model")
        terminal_frontend.terminal.assistant_markdown.reset()

    def tearDown(self):
        terminal_frontend.terminal.assistant_markdown.reset()
        loki.current_session().runtime_config = self._old_config

    def test_streamed_events_render_styled_and_not_duplicated(self):
        transcript = [formats.message_item("user", "hello")]
        events = []
        reply = "prefix **hello** `world`"

        async def chat_fn(
                items, on_text_delta, *, codex_turn_state):
            for chunk in ["prefix **hel", "lo** `wor", "ld`"]:
                on_text_delta(chunk)
            return [formats.message_item("assistant", reply)]

        result = asyncio.run(loki.run_tool_loop_async(
            transcript,
            chat_fn=chat_fn,
            on_event=events.append,
            stream_chat=True,
        ))

        self.assertEqual(result, reply)
        self.assertEqual(
            [event["type"] for event in events],
            [
                "assistant_start",
                "assistant_delta",
                "assistant_delta",
                "assistant_delta",
                "assistant_end",
            ],
        )

        output = io.StringIO()
        with mock.patch.object(
                terminal_frontend, "terminal", StyledTerminal()), \
                contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(io.StringIO()):
            for event in events:
                terminal_frontend._terminal_agent_event(event)

        self.assertEqual(
            output.getvalue(),
            f"\nlocal-model: prefix {BOLD}hello{RESET} "
            f"{CODE}world{RESET}\n",
        )


class TerminalDiagnosticTests(unittest.TestCase):
    def test_hook_stderr_stays_multiline_but_neutralizes_controls(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with mock.patch.object(
                terminal_frontend, "terminal", StyledTerminal()), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            terminal_frontend._report_hook_stderr(
                ["hook\x1b]0;name\x07"],
                "first\x1b]0;owned\x07\n"
                "second\u009b \u6a21\u578b",
            )

        self.assertEqual(
            stderr.getvalue(),
            "Hook 'hook\\x1b]0;name\\x07' stderr:\n"
            "first^[]0;owned^G\nsecond\\x9b \u6a21\u578b\n",
        )


if __name__ == "__main__":
    unittest.main()
