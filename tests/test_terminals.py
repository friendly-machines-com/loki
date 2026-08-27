import asyncio
import contextlib
import io
import pathlib
import sys
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from loki_agent import terminals


def feed_bytes(reader, data):
    for byte in data:
        reader._feed_byte(byte)
    events = list(reader.pending)
    reader.pending.clear()
    return events


class TerminalResourceSafetyTests(unittest.TestCase):
    def test_terminal_mode_rolls_back_when_mode_change_fails(self):
        cc = [b"\0"] * (
            max(terminals.termios.VMIN, terminals.termios.VTIME) + 1)
        old_attrs = [
            getattr(terminals.termios, "IXON", 0),
            0,
            0,
            terminals.termios.ICANON
            | terminals.termios.ECHO
            | terminals.termios.ISIG,
            0,
            0,
            cc,
        ]
        applied_attrs = []

        def fake_tcsetattr(fd, when, attrs):
            applied_attrs.append(attrs)
            if len(applied_attrs) == 1:
                raise OSError("mode change failed")

        mode = terminals.TerminalMode(123, enabled=True)
        with (
            mock.patch.object(
                terminals.termios, "tcgetattr", return_value=old_attrs),
            mock.patch.object(
                terminals.termios, "tcsetattr",
                side_effect=fake_tcsetattr),
            self.assertRaisesRegex(OSError, "mode change failed"),
        ):
            mode.__enter__()

        self.assertEqual(len(applied_attrs), 2)
        self.assertEqual(applied_attrs[0][0], old_attrs[0])
        self.assertEqual(applied_attrs[1], old_attrs)
        self.assertIsNone(mode.old_attrs)

    def test_byte_reader_restores_flags_when_registration_fails(self):
        old_flags = 0x10
        written_flags = []

        class FailingLoop:
            def add_reader(self, fd, callback):
                raise RuntimeError("reader registration failed")

        def fake_fcntl(fd, command, value=None):
            if command == terminals.fcntl.F_GETFL:
                return old_flags
            self.assertEqual(command, terminals.fcntl.F_SETFL)
            written_flags.append(value)

        async def exercise():
            reader = terminals.AsyncByteReader(123)
            with (
                mock.patch.object(
                    terminals.asyncio, "get_running_loop",
                    return_value=FailingLoop()),
                mock.patch.object(
                    terminals.fcntl, "fcntl", side_effect=fake_fcntl),
            ):
                with self.assertRaisesRegex(
                        RuntimeError, "registration failed"):
                    await reader.__aenter__()
            self.assertIsNone(reader.loop)
            self.assertIsNone(reader.old_flags)

        asyncio.run(exercise())

        self.assertEqual(
            written_flags,
            [old_flags | terminals.os.O_NONBLOCK, old_flags],
        )

    def test_byte_reader_restores_flags_when_removal_fails(self):
        old_flags = 0x10
        written_flags = []

        class FailingLoop:
            def remove_reader(self, fd):
                raise RuntimeError("reader removal failed")

        async def exercise():
            reader = terminals.AsyncByteReader(123)
            reader.loop = FailingLoop()
            reader.old_flags = old_flags
            reader._reader_registered = True
            with mock.patch.object(
                    terminals.fcntl, "fcntl",
                    side_effect=lambda fd, command, value: (
                        written_flags.append(value))):
                with self.assertRaisesRegex(RuntimeError, "removal failed"):
                    await reader.__aexit__(None, None, None)
            self.assertIsNone(reader.loop)
            self.assertIsNone(reader.old_flags)

        asyncio.run(exercise())

        self.assertEqual(written_flags, [old_flags])

    def test_key_reader_rolls_back_byte_reader_after_resize_setup_error(self):
        calls = []

        class RecordingByteReader:
            async def __aenter__(self):
                calls.append("byte enter")
                return self

            async def __aexit__(self, exc_type, exc, tb):
                calls.append("byte exit")

        class FailingLoop:
            def add_signal_handler(self, signum, callback):
                raise OSError("resize registration failed")

        async def exercise():
            reader = terminals.AsyncKeyReader(123, watch_resize=True)
            reader.byte_reader = RecordingByteReader()
            with mock.patch.object(
                    terminals.asyncio, "get_running_loop",
                    return_value=FailingLoop()):
                with self.assertRaisesRegex(
                        OSError, "resize registration failed"):
                    await reader.__aenter__()
            self.assertIsNone(reader.loop)

        asyncio.run(exercise())

        self.assertEqual(calls, ["byte enter", "byte exit"])

    def test_input_session_restores_mode_when_reader_setup_fails(self):
        calls = []

        class RecordingMode:
            def __init__(self, fd, enabled):
                pass

            def __enter__(self):
                calls.append("mode enter")
                return self

            def __exit__(self, exc_type, exc, tb):
                calls.append("mode exit")

        class FailingReader:
            async def __aenter__(self):
                calls.append("reader enter")
                raise RuntimeError("reader setup failed")

            async def __aexit__(self, exc_type, exc, tb):
                calls.append("reader exit")

        async def exercise():
            session = terminals.InputSession(fd=123)
            session.interactive = True
            session.reader = FailingReader()
            with (
                mock.patch.object(
                    terminals, "TerminalMode", RecordingMode),
                self.assertRaisesRegex(RuntimeError, "reader setup failed"),
            ):
                await session.__aenter__()
            self.assertIsNone(session._mode)
            self.assertIsNone(session._resources)

        asyncio.run(exercise())

        self.assertEqual(calls, ["mode enter", "reader enter", "mode exit"])

    def test_input_session_restores_mode_when_reader_cleanup_fails(self):
        calls = []

        class RecordingMode:
            def __init__(self, fd, enabled):
                pass

            def __enter__(self):
                calls.append("mode enter")
                return self

            def __exit__(self, exc_type, exc, tb):
                calls.append("mode exit")

        class FailingReader:
            async def __aenter__(self):
                calls.append("reader enter")
                return self

            async def __aexit__(self, exc_type, exc, tb):
                calls.append("reader exit")
                raise RuntimeError("reader cleanup failed")

        async def idle_producer():
            await asyncio.Event().wait()

        async def exercise():
            session = terminals.InputSession(fd=123)
            session.interactive = True
            session.reader = FailingReader()
            session._produce = idle_producer
            with mock.patch.object(
                    terminals, "TerminalMode", RecordingMode):
                await session.__aenter__()
                with self.assertRaisesRegex(
                        RuntimeError, "reader cleanup failed"):
                    await session.__aexit__(None, None, None)
            self.assertIsNone(session._mode)
            self.assertIsNone(session._resources)

        asyncio.run(exercise())

        self.assertEqual(
            calls,
            ["mode enter", "reader enter", "reader exit", "mode exit"],
        )


class RecordingTerminal:
    def __init__(self):
        self.calls = []

    def set_clipping_region(self, first_row, last_row):
        self.calls.append(("set_clipping_region", first_row, last_row))

    def goto_position(self, row, column):
        self.calls.append(("goto_position", row, column))

    def set_background_color(self, index):
        self.calls.append(("set_background_color", index))

    def set_reverse_video(self, enabled):
        self.calls.append(("set_reverse_video", enabled))

    def clear_to_end_of_screen(self):
        self.calls.append(("clear_to_end_of_screen",))

    def save_cursor_position(self):
        self.calls.append(("save_cursor_position",))

    def restore_cursor_position(self):
        self.calls.append(("restore_cursor_position",))

    def reset_colors_and_flags(self):
        self.calls.append(("reset_colors_and_flags",))

    def flush(self):
        self.calls.append(("flush",))


class TerminfoKeySequenceTests(unittest.TestCase):
    def test_import_failure_returns_no_sequences(self):
        original_import = __import__

        def import_without_curses(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("curses unavailable")
            return original_import(name, *args, **kwargs)

        with mock.patch(
                "builtins.__import__", side_effect=import_without_curses):
            sequences = terminals.terminfo_key_sequences(output_fd=7)

        self.assertEqual(sequences, {})

    def test_setup_failure_returns_no_sequences(self):
        fake_curses = mock.Mock()
        fake_curses.setupterm.side_effect = RuntimeError(
            "terminfo unavailable")

        with mock.patch.dict(sys.modules, {"curses": fake_curses}):
            sequences = terminals.terminfo_key_sequences(output_fd=7)

        self.assertEqual(sequences, {})
        fake_curses.setupterm.assert_called_once_with(fd=7)
        fake_curses.tigetstr.assert_not_called()

    def test_lookup_is_best_effort_and_rejects_unsafe_values(self):
        values = {
            "kcuu1": b"\x1b[9A",
            "kcud1": None,
            "kcuf1": "not bytes",
            "kcub1": b"D",
            "khome": b"\x1b" + b"x" * terminals.MAX_KEY_SEQUENCE_BYTES,
            "kend": b"\x1b[99~",
            "kpp": b"\x1b[77~",
        }
        fake_curses = mock.Mock()

        def lookup(capability):
            if capability == "kdch1":
                raise RuntimeError("one broken capability")
            return values.get(capability)

        fake_curses.tigetstr.side_effect = lookup
        with mock.patch.dict(sys.modules, {"curses": fake_curses}):
            sequences = terminals.terminfo_key_sequences(output_fd=7)

        self.assertEqual(
            sequences,
            {
                b"\x1b[9A": "CURSOR_UP",
                b"\x1b[99~": "END",
                b"\x1b[77~": "PAGE_UP",
            },
        )
        fake_curses.setupterm.assert_called_once_with(fd=7)
        self.assertEqual(
            fake_curses.tigetstr.call_count,
            len(terminals.TERMINFO_KEY_CAPABILITIES),
        )
        fake_curses.initscr.assert_not_called()
        fake_curses.newterm.assert_not_called()
        fake_curses.raw.assert_not_called()
        fake_curses.cbreak.assert_not_called()
        fake_curses.putp.assert_not_called()

    def test_reader_uses_terminfo_additively(self):
        optional_sequences = {
            b"\x1b[A": "DELETE",
            b"\x1b[99~": "HOME",
            b"\x1bOA": "CURSOR_UP",
        }
        with mock.patch.object(
                terminals, "terminfo_key_sequences",
                return_value=optional_sequences) as lookup:
            reader = terminals.AsyncKeyReader(
                fd=0, use_terminfo=True, output_fd=7)

        lookup.assert_called_once_with(7)
        self.assertEqual(
            feed_bytes(reader, b"\x1b[A"),
            [terminals.KeyEvent("CURSOR_UP")],
        )
        self.assertEqual(
            feed_bytes(reader, b"\x1b[99~"),
            [terminals.KeyEvent("HOME")],
        )
        self.assertEqual(
            feed_bytes(reader, b"\x1bOA"),
            [terminals.KeyEvent("CURSOR_UP")],
        )

    def test_reader_does_not_lookup_terminfo_unless_enabled(self):
        with mock.patch.object(
                terminals, "terminfo_key_sequences") as lookup:
            terminals.AsyncKeyReader(fd=0)

        lookup.assert_not_called()


class AsyncKeyReaderTests(unittest.TestCase):
    def tty_attrs(
            self, erase=b"\x7f", word_erase=b"\x17",
            interrupt=b"\x03"):
        cc = [b"\0"] * (
            max(
                terminals.termios.VERASE,
                terminals.termios.VWERASE,
                terminals.termios.VINTR,
                terminals.termios.VMIN,
                terminals.termios.VTIME,
            ) + 1
        )
        cc[terminals.termios.VERASE] = erase
        cc[terminals.termios.VWERASE] = word_erase
        cc[terminals.termios.VINTR] = interrupt
        return [0, 0, 0, 0, 0, 0, cc]

    def test_decodes_text_ascii_and_utf8(self):
        reader = terminals.AsyncKeyReader(fd=0)

        self.assertEqual(feed_bytes(reader, b"a"), [terminals.KeyEvent("TEXT", "a")])
        self.assertEqual(feed_bytes(reader, bytes([0xc3])), [])
        self.assertEqual(feed_bytes(reader, bytes([0xa9])), [terminals.KeyEvent("TEXT", "\u00e9")])

    def test_decodes_control_keys(self):
        control_bytes = (
            terminals.FALLBACK_BACKSPACE_BYTES,
            terminals.FALLBACK_BACKSPACE_WORD_BYTES,
            terminals.FALLBACK_INTERRUPT_BYTES,
        )
        with mock.patch.object(
                terminals, "terminal_control_bytes",
                return_value=control_bytes):
            reader = terminals.AsyncKeyReader(fd=0)

        events = feed_bytes(reader, b"\x03\x04\r\n\x7f\x08\x17")

        self.assertEqual(
            events,
            [
                terminals.KeyEvent("CTRL_C"),
                terminals.KeyEvent("CTRL_D"),
                terminals.KeyEvent("ENTER"),
                terminals.KeyEvent("ENTER"),
                terminals.KeyEvent("BACKSPACE"),
                terminals.KeyEvent("BACKSPACE"),
                terminals.KeyEvent("BACKSPACE_WORD"),
            ],
        )

    def test_uses_configured_interrupt_byte(self):
        attrs = self.tty_attrs(interrupt=b"\x18")
        with (
            mock.patch.object(
                terminals.termios, "tcgetattr", return_value=attrs),
            mock.patch.object(
                terminals.os, "fpathconf", return_value=0xff),
        ):
            reader = terminals.AsyncKeyReader(fd=123)

        self.assertEqual(
            feed_bytes(reader, b"\x18"),
            [terminals.KeyEvent("CTRL_C")],
        )
        self.assertTrue(reader.cancel_requested)
        self.assertTrue(reader.cancel_event.is_set())

    def test_uses_configured_character_and_word_erase_bytes(self):
        cases = [
            (b"\x08", b"\x17"),
            (b"\x7f", b"\x17"),
            (b"\x15", b"\x16"),
            (0x7f, 0x17),
        ]
        for erase, word_erase in cases:
            with self.subTest(erase=erase, word_erase=word_erase):
                attrs = self.tty_attrs(erase, word_erase)
                with (
                    mock.patch.object(
                        terminals.termios, "tcgetattr", return_value=attrs),
                    mock.patch.object(
                        terminals.os, "fpathconf", return_value=0xff),
                ):
                    reader = terminals.AsyncKeyReader(fd=123)

                erase_byte = (
                    erase if isinstance(erase, int) else erase[0])
                word_erase_byte = (
                    word_erase
                    if isinstance(word_erase, int)
                    else word_erase[0]
                )
                self.assertEqual(
                    feed_bytes(
                        reader, bytes((erase_byte, word_erase_byte))),
                    [
                        terminals.KeyEvent("BACKSPACE"),
                        terminals.KeyEvent("BACKSPACE_WORD"),
                    ],
                )

    def test_character_erase_wins_if_both_values_are_equal(self):
        attrs = self.tty_attrs(b"\x15", b"\x15")
        with (
            mock.patch.object(
                terminals.termios, "tcgetattr", return_value=attrs),
            mock.patch.object(
                terminals.os, "fpathconf", return_value=0xff),
        ):
            reader = terminals.AsyncKeyReader(fd=123)

        self.assertEqual(
            feed_bytes(reader, b"\x15"),
            [terminals.KeyEvent("BACKSPACE")],
        )

    def test_invalid_or_disabled_values_use_independent_fallbacks(self):
        attrs = self.tty_attrs(
            b"invalid", b"\x15", interrupt=b"\x15")
        with (
            mock.patch.object(
                terminals.termios, "tcgetattr", return_value=attrs),
            mock.patch.object(
                terminals.os, "fpathconf", return_value=0x15),
        ):
            reader = terminals.AsyncKeyReader(fd=123)

        self.assertEqual(
            feed_bytes(reader, b"\x08\x7f\x17\x03"),
            [
                terminals.KeyEvent("BACKSPACE"),
                terminals.KeyEvent("BACKSPACE"),
                terminals.KeyEvent("BACKSPACE_WORD"),
                terminals.KeyEvent("CTRL_C"),
            ],
        )

    def test_termios_error_uses_control_fallbacks(self):
        error = terminals.termios.error(25, "not a tty")
        with mock.patch.object(
                terminals.termios, "tcgetattr", side_effect=error):
            reader = terminals.AsyncKeyReader(fd=123)

        self.assertEqual(
            feed_bytes(reader, b"\x08\x7f\x17\x03"),
            [
                terminals.KeyEvent("BACKSPACE"),
                terminals.KeyEvent("BACKSPACE"),
                terminals.KeyEvent("BACKSPACE_WORD"),
                terminals.KeyEvent("CTRL_C"),
            ],
        )

    def test_decodes_known_escape_sequences(self):
        reader = terminals.AsyncKeyReader(fd=0)

        self.assertEqual(feed_bytes(reader, b"\x1b[A"), [terminals.KeyEvent("CURSOR_UP")])
        self.assertEqual(feed_bytes(reader, b"\x1b[B"), [terminals.KeyEvent("CURSOR_DOWN")])
        self.assertEqual(feed_bytes(reader, b"\x1b[C"), [terminals.KeyEvent("CURSOR_RIGHT")])
        self.assertEqual(feed_bytes(reader, b"\x1b[D"), [terminals.KeyEvent("CURSOR_LEFT")])
        self.assertEqual(feed_bytes(reader, b"\x1b[H"), [terminals.KeyEvent("HOME")])
        self.assertEqual(feed_bytes(reader, b"\x1b[F"), [terminals.KeyEvent("END")])
        self.assertEqual(feed_bytes(reader, b"\x1b[3~"), [terminals.KeyEvent("DELETE")])
        self.assertEqual(feed_bytes(reader, b"\x1b[5~"), [terminals.KeyEvent("PAGE_UP")])
        self.assertEqual(feed_bytes(reader, b"\x1b[6~"), [terminals.KeyEvent("PAGE_DOWN")])

    def test_cpr_requires_exact_two_numeric_parameters(self):
        reader = terminals.AsyncKeyReader(fd=0)

        self.assertEqual(
            feed_bytes(reader, b"\x1b[12;34R"),
            [terminals.KeyEvent("CPR", "\x1b[12;34R")],
        )
        self.assertEqual(feed_bytes(reader, b"\x1b[12;34;56R"), [])

    def test_bracketed_paste_is_one_text_event_with_newlines(self):
        reader = terminals.AsyncKeyReader(fd=0)

        events = feed_bytes(reader, b"\x1b[200~a\nb\x1b[201~")

        self.assertEqual(
            events,
            [
                terminals.KeyEvent("PASTE_START"),
                terminals.KeyEvent("TEXT", "a\nb"),
                terminals.KeyEvent("PASTE_END"),
            ],
        )
        self.assertFalse(reader.paste_mode)

    def test_bracketed_paste_coalesces_across_read_chunks(self):
        class ChunkByteReader:
            def __init__(self):
                self.chunks = [
                    b"\x1b[200~caf\xc3",
                    b"\xa9\nsecond",
                    b" line\x1b[201~",
                ]

            async def read(self):
                return self.chunks.pop(0)

        async def scenario():
            reader = terminals.AsyncKeyReader(fd=0)
            reader.byte_reader = ChunkByteReader()
            return [await reader.read_key() for _ in range(3)]

        self.assertEqual(
            asyncio.run(scenario()),
            [
                terminals.KeyEvent("PASTE_START"),
                terminals.KeyEvent("TEXT", "caf\u00e9\nsecond line"),
                terminals.KeyEvent("PASTE_END"),
            ],
        )

    def test_eof_finishes_an_incomplete_bracketed_paste(self):
        class ChunkByteReader:
            def __init__(self):
                self.chunks = [b"\x1b[200~partial", b""]

            async def read(self):
                return self.chunks.pop(0)

        async def scenario():
            reader = terminals.AsyncKeyReader(fd=0)
            reader.byte_reader = ChunkByteReader()
            return [await reader.read_key() for _ in range(4)]

        self.assertEqual(
            asyncio.run(scenario()),
            [
                terminals.KeyEvent("PASTE_START"),
                terminals.KeyEvent("TEXT", "partial"),
                terminals.KeyEvent("PASTE_END"),
                terminals.KeyEvent("EOF"),
            ],
        )

    def test_unknown_short_escape_sequence_is_ignored(self):
        reader = terminals.AsyncKeyReader(fd=0)

        self.assertEqual(feed_bytes(reader, b"\x1bX"), [])
        self.assertEqual(reader.escape, bytearray())

    def test_read_key_returns_eof_for_empty_chunk_without_pending_events(self):
        class EmptyByteReader:
            async def read(self):
                return b""

        reader = terminals.AsyncKeyReader(fd=0)
        reader.byte_reader = EmptyByteReader()

        event = asyncio.run(reader.read_key())

        self.assertEqual(event, terminals.KeyEvent("EOF"))

    def test_resize_wakeup_is_not_mistaken_for_eof(self):
        class WakeByteReader:
            def __init__(self):
                self.queue = asyncio.Queue()

            async def read(self):
                return await self.queue.get()

        async def scenario():
            reader = terminals.AsyncKeyReader(fd=0)
            reader.byte_reader = WakeByteReader()
            reader._on_resize()
            resize = await reader.read_key()
            reader.byte_reader.queue.put_nowait(b"x")
            text = await reader.read_key()
            return resize, text

        self.assertEqual(
            asyncio.run(scenario()),
            (
                terminals.KeyEvent("RESIZE"),
                terminals.KeyEvent("TEXT", "x"),
            ),
        )


class UserMessageQueueTests(unittest.TestCase):
    def test_reports_non_sentinel_messages_on_enqueue_and_dequeue(self):
        counts = []
        queue = terminals.UserMessageQueue(counts.append)

        queue.put_nowait("first")
        queue.put_nowait("")
        queue.put_nowait(None)

        self.assertEqual(queue.message_count, 2)
        self.assertEqual(counts, [1, 2])
        self.assertEqual(queue.get_nowait(), "first")
        self.assertEqual(queue.get_nowait(), "")
        self.assertIsNone(queue.get_nowait())
        self.assertEqual(queue.message_count, 0)
        self.assertEqual(counts, [1, 2, 1, 0])

    def test_discard_resets_count_and_removes_pending_items(self):
        counts = []
        queue = terminals.UserMessageQueue(counts.append)
        queue.put_nowait("first")
        queue.put_nowait("second")
        queue.put_nowait(None)

        queue.discard_pending_messages()

        self.assertTrue(queue.empty())
        self.assertEqual(queue.message_count, 0)
        self.assertEqual(counts, [1, 2, 0])


class InputBufferTests(unittest.TestCase):
    def test_word_left_on_empty_buffer_keeps_cursor_valid(self):
        buffer = terminals.InputBuffer()
        buffer.word_left()
        buffer.insert("a")
        buffer.insert("b")
        self.assertEqual(buffer.cursor, 2)
        self.assertEqual(buffer.text(), "ab")

    def test_insert_and_cursor_editing(self):
        buffer = terminals.InputBuffer()

        buffer.insert("abc")
        buffer.left()
        buffer.left()
        buffer.insert("X")

        self.assertEqual(buffer.text(), "aXbc")
        self.assertEqual(buffer.before_cursor(), "aX")
        self.assertEqual(buffer.after_cursor(), "bc")

        buffer.backspace()
        self.assertEqual(buffer.text(), "abc")
        self.assertEqual(buffer.before_cursor(), "a")

        buffer.delete()
        self.assertEqual(buffer.text(), "ac")

        buffer.home()
        buffer.backspace()
        self.assertEqual(buffer.text(), "ac")

        buffer.end()
        buffer.delete()
        self.assertEqual(buffer.text(), "ac")

    def test_bulk_insert_shifts_the_suffix_once_and_updates_cursor(self):
        buffer = terminals.InputBuffer()
        buffer.insert("prefix--suffix")
        buffer.cursor = len("prefix")

        buffer.insert("a large pasted block")

        self.assertEqual(
            buffer.text(), "prefixa large pasted block--suffix")
        self.assertEqual(buffer.cursor, len("prefixa large pasted block"))

    def test_backspace_word_deletes_exactly_to_ctrl_left_boundary(self):
        cases = [
            ("alpha beta", len("alpha beta")),
            ("alpha beta tail", len("alpha beta ")),
            ("alpha   ", len("alpha   ")),
            ("", 0),
        ]
        for text, cursor in cases:
            with self.subTest(text=text, cursor=cursor):
                movement = terminals.InputBuffer()
                movement.insert(text)
                movement.cursor = cursor
                movement.word_left()

                deletion = terminals.InputBuffer()
                deletion.insert(text)
                deletion.cursor = cursor
                deletion.backspace_word()

                self.assertEqual(deletion.cursor, movement.cursor)
                self.assertEqual(
                    deletion.text(),
                    text[:movement.cursor] + text[cursor:],
                )


class PromptControllerTests(unittest.TestCase):
    def read_with_events(self, events, history=None):
        old_stdin = sys.stdin
        old_isatty = terminals.os.isatty
        old_key_reader = terminals.AsyncKeyReader
        old_terminal_mode = terminals.TerminalMode

        class FakeStdin:
            def fileno(self):
                return 0

        class FakeTerminalMode:
            def __init__(self, fd, enabled):
                self.fd = fd
                self.enabled = enabled

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeKeyReader:
            def __init__(self, fd, watch_resize=False, **kwargs):
                self.events = list(events)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def read_key(self):
                if self.events:
                    return self.events.pop(0)
                return terminals.KeyEvent("EOF")

        try:
            sys.stdin = FakeStdin()
            terminals.os.isatty = lambda fd: False
            terminals.AsyncKeyReader = FakeKeyReader
            terminals.TerminalMode = FakeTerminalMode
            controller = terminals.PromptController(RecordingTerminal(), history=history)
            return asyncio.run(controller.read_text())
        finally:
            sys.stdin = old_stdin
            terminals.os.isatty = old_isatty
            terminals.AsyncKeyReader = old_key_reader
            terminals.TerminalMode = old_terminal_mode

    def test_read_text_applies_keyboard_editing(self):
        result = self.read_with_events(
            [
                terminals.KeyEvent("TEXT", "abc"),
                terminals.KeyEvent("CURSOR_LEFT"),
                terminals.KeyEvent("CURSOR_LEFT"),
                terminals.KeyEvent("TEXT", "X"),
                terminals.KeyEvent("END"),
                terminals.KeyEvent("TEXT", "!"),
                terminals.KeyEvent("ENTER"),
            ]
        )

        self.assertEqual(result, "aXbc!")

    def test_reader_configuration_precedes_tty_mode_change(self):
        calls = []

        class TtyOutput(io.StringIO):
            def fileno(self):
                return 1

        class RecordingReader:
            def __init__(
                    self, fd, watch_resize=False, use_terminfo=False,
                    output_fd=None):
                calls.append(
                    ("reader", watch_resize, use_terminfo, output_fd))

            async def __aenter__(self):
                calls.append("reader enter")
                return self

            async def __aexit__(self, exc_type, exc, tb):
                calls.append("reader exit")

            async def read_key(self):
                return terminals.KeyEvent("ENTER")

        class RecordingMode:
            def __init__(self, fd, enabled):
                pass

            def __enter__(self):
                calls.append("mode enter")
                return self

            def __exit__(self, exc_type, exc, tb):
                calls.append("mode exit")

        with (
            mock.patch.object(terminals, "new_stdin", 123),
            mock.patch.object(terminals.os, "isatty", return_value=True),
            mock.patch.object(terminals, "AsyncKeyReader", RecordingReader),
            mock.patch.object(terminals, "TerminalMode", RecordingMode),
            contextlib.redirect_stdout(TtyOutput()),
        ):
            result = asyncio.run(
                terminals.PromptController(RecordingTerminal()).read_text())

        self.assertEqual(result, "")
        self.assertEqual(calls[0][0:3], ("reader", True, True))
        self.assertEqual(
            calls[1:],
            ["mode enter", "reader enter", "reader exit", "mode exit"],
        )

    def test_read_text_applies_ctrl_backspace_after_ctrl_left(self):
        result = self.read_with_events(
            [
                terminals.KeyEvent("TEXT", "alpha beta tail"),
                terminals.KeyEvent("CURSOR_WORD_LEFT"),
                terminals.KeyEvent("BACKSPACE_WORD"),
                terminals.KeyEvent("ENTER"),
            ]
        )

        self.assertEqual(result, "alpha tail")

    def test_bracketed_paste_redraws_once_at_paste_end(self):
        pasted = "x" * 10000

        class FakeReader:
            def __init__(self):
                self.events = iter([
                    terminals.KeyEvent("PASTE_START"),
                    terminals.KeyEvent("TEXT", pasted),
                    terminals.KeyEvent("RESIZE"),
                    terminals.KeyEvent("PASTE_END"),
                    terminals.KeyEvent("ENTER"),
                ])

            async def read_key(self):
                return next(self.events)

        class RecordingRenderer:
            def __init__(self):
                self.rendered = []

            def render(self, buffer):
                self.rendered.append(buffer.text())

        async def scenario():
            controller = terminals.PromptController(RecordingTerminal())
            renderer = RecordingRenderer()
            result = await controller._read_text_from_reader(
                FakeReader(),
                True,
                terminals.InputBuffer(),
                renderer,
                1,
                1,
                0,
                "",
            )
            return result, renderer.rendered

        result, rendered = asyncio.run(scenario())

        self.assertEqual(result, pasted)
        self.assertEqual(rendered, ["", pasted])

    def test_read_text_navigates_history_and_restores_saved_input(self):
        result = self.read_with_events(
            [
                terminals.KeyEvent("TEXT", "draft"),
                terminals.KeyEvent("CURSOR_UP"),
                terminals.KeyEvent("CURSOR_DOWN"),
                terminals.KeyEvent("ENTER"),
            ],
            history=["old1", "old2"],
        )

        self.assertEqual(result, "draft")

    def test_read_text_returns_history_selection(self):
        result = self.read_with_events(
            [
                terminals.KeyEvent("CURSOR_UP"),
                terminals.KeyEvent("CURSOR_UP"),
                terminals.KeyEvent("CURSOR_DOWN"),
                terminals.KeyEvent("ENTER"),
            ],
            history=["old1", "old2"],
        )

        self.assertEqual(result, "old2")

    def test_eof_and_ctrl_d_return_partial_buffer(self):
        self.assertEqual(
            self.read_with_events([terminals.KeyEvent("TEXT", "abc"), terminals.KeyEvent("EOF")]),
            "abc",
        )
        self.assertEqual(
            self.read_with_events([terminals.KeyEvent("TEXT", "abc"), terminals.KeyEvent("CTRL_D")]),
            "abc",
        )

    def test_eof_ctrl_d_and_ctrl_c_on_empty_buffer_raise(self):
        with self.assertRaises(EOFError):
            self.read_with_events([terminals.KeyEvent("EOF")])
        with self.assertRaises(EOFError):
            self.read_with_events([terminals.KeyEvent("CTRL_D")])
        with self.assertRaises(KeyboardInterrupt):
            self.read_with_events([terminals.KeyEvent("CTRL_C")])


class InputModalTests(unittest.TestCase):
    def test_modal_is_the_exclusive_direct_input_path(self):
        session = terminals.InputSession(fd=0)
        calls = []

        async def pause():
            calls.append("pause")

        async def resume():
            calls.append("resume")

        async def fake_get_input_async(
                prompt=None, history=None, session=None, on_mode_cycle=None):
            calls.append(("prompt", prompt, history, session))
            return "answer"

        session._pause = pause
        session._resume = resume
        old_get_input_async = terminals.get_input_async
        terminals.get_input_async = fake_get_input_async

        async def exercise():
            modal = session.modal()
            with self.assertRaisesRegex(RuntimeError, "outside"):
                await modal.prompt("Question: ")
            async with modal:
                self.assertIs(session._modal, modal)
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    async with session.modal():
                        pass
                return await modal.prompt("Question: ", ["old"])

        try:
            result = asyncio.run(exercise())
        finally:
            terminals.get_input_async = old_get_input_async

        self.assertEqual(result, "answer")
        self.assertIsNone(session._modal)
        self.assertEqual(calls, [
            "pause",
            ("prompt", "Question: ", ["old"], session.reader),
            "resume",
        ])

    def test_modal_restores_normal_input_after_exception(self):
        session = terminals.InputSession(fd=0)
        calls = []

        async def pause():
            calls.append("pause")

        async def resume():
            calls.append("resume")

        session._pause = pause
        session._resume = resume

        async def exercise():
            with self.assertRaisesRegex(ValueError, "boom"):
                async with session.modal():
                    raise ValueError("boom")

        asyncio.run(exercise())

        self.assertEqual(calls, ["pause", "resume"])
        self.assertIsNone(session._modal)


class PromptRendererTests(unittest.TestCase):
    def test_render_refreshes_input_area_status_bar_and_cursor_position(self):
        recorder = RecordingTerminal()
        buffer = terminals.InputBuffer()
        buffer.insert("abc")
        buffer.left()

        old_refresh = terminals.refresh_terminal_layout
        old_update_status_bar = terminals.update_status_bar
        old_input_area = terminals.input_area
        try:
            terminals.refresh_terminal_layout = lambda: None
            terminals.update_status_bar = lambda: recorder.calls.append(("update_status_bar",))
            terminals.input_area = (10, 13)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                terminals.PromptRenderer(recorder, "User: ").render(buffer)
        finally:
            terminals.refresh_terminal_layout = old_refresh
            terminals.update_status_bar = old_update_status_bar
            terminals.input_area = old_input_area

        self.assertEqual(out.getvalue(), "User: abc")
        self.assertEqual(
            recorder.calls,
            [
                ("save_cursor_position",),
                ("set_clipping_region", 10, 13),
                ("goto_position", 1, 1),
                ("set_background_color", terminals.INPUT_COLOR),
                ("clear_to_end_of_screen",),
                ("update_status_bar",),
                ("set_clipping_region", 10, 13),
                ("goto_position", 1, 1),
                ("set_background_color", terminals.INPUT_COLOR),
                ("set_reverse_video", True),
                ("set_reverse_video", False),
                ("set_clipping_region", *terminals.output_area),
                ("restore_cursor_position",),
                ("reset_colors_and_flags",),
                ("flush",),
            ],
        )


class RestoreOutputAreaTests(unittest.TestCase):
    def test_restore_output_area_resets_colors_and_flushes(self):
        old_terminal = terminals.terminal
        recorder = RecordingTerminal()
        try:
            terminals.terminal = recorder
            terminals.restore_output_area_after_input()
        finally:
            terminals.terminal = old_terminal

        self.assertEqual(recorder.calls, [("reset_colors_and_flags",), ("flush",)])


if __name__ == "__main__":
    unittest.main()
