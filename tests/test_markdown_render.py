"""Bounded incremental Markdown rendering and terminal-event wiring."""

import asyncio
import contextlib
import io
import os
import pathlib
import random
import sys
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("LOKI_API_KEY", "test-key")
os.environ.setdefault("LOKI_API_BASE", "https://api.openai.com/v1/responses")
os.environ.setdefault("LOKI_PROVIDER", "openai_responses")

from loki_agent import formats
from loki_agent import loki
from loki_agent import terminals


BOLD = "\033[1m"
ITALIC = "\033[3m"
CODE = "\033[36m"
RESET = "\033[0m"


class StyledTerminal:
    def markdown_to_ansi(self, text):
        return terminals.markdown_line_to_ansi(text)


class NoneTerminal:
    def markdown_to_ansi(self, text):
        return None


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


class DifferentialTests(unittest.TestCase):
    """Chunk divisions cannot change the final rendering."""

    @staticmethod
    def stream_concat(text, cuts, *, max_unresolved=4096):
        renderer = terminals.BoundedMarkdownAnsi(
            style=True, max_unresolved=max_unresolved)
        output = []
        for chunk in split_at(text, cuts):
            fragments = renderer.feed(chunk)
            for fragment in fragments:
                opens = (
                    fragment.count(BOLD)
                    + fragment.count(ITALIC)
                    + fragment.count(CODE)
                )
                if fragment.count(RESET) != opens:
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
        self._old_model = loki.model
        loki.model = "local-model"
        loki.terminal.assistant_markdown.reset()

    def tearDown(self):
        loki.terminal.assistant_markdown.reset()
        loki.model = self._old_model

    def replay(self, events, terminal_stub):
        output = io.StringIO()
        with mock.patch.object(terminals, "terminal", terminal_stub), \
                contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(io.StringIO()):
            for event in events:
                loki._terminal_agent_event(event)
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
        with mock.patch.object(terminals, "terminal", StyledTerminal()), \
                contextlib.redirect_stdout(output):
            loki._terminal_agent_event({"type": "assistant_start"})
            loki._terminal_agent_event({
                "type": "assistant_delta",
                "content": "visible without a newline",
            })
            before_end = output.getvalue()
            loki._terminal_agent_event({
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
        self._old_model = loki.model
        loki.model = "local-model"
        loki.terminal.assistant_markdown.reset()

    def tearDown(self):
        loki.terminal.assistant_markdown.reset()
        loki.model = self._old_model

    def test_streamed_events_render_styled_and_not_duplicated(self):
        transcript = [formats.message_item("user", "hello")]
        events = []
        reply = "prefix **hello** `world`"

        async def chat_fn(items, on_text_delta):
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
        with mock.patch.object(terminals, "terminal", StyledTerminal()), \
                contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(io.StringIO()):
            for event in events:
                loki._terminal_agent_event(event)

        self.assertEqual(
            output.getvalue(),
            f"\nlocal-model: prefix {BOLD}hello{RESET} "
            f"{CODE}world{RESET}\n",
        )


if __name__ == "__main__":
    unittest.main()
