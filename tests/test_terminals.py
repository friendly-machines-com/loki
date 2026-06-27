import asyncio
import contextlib
import io
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from loki_agent import terminals


def feed_bytes(reader, data):
    for byte in data:
        reader._feed_byte(byte)
    events = list(reader.pending)
    reader.pending.clear()
    return events


class RecordingTerminal:
    def __init__(self):
        self.calls = []

    def set_clipping_region(self, first_row, last_row):
        self.calls.append(("set_clipping_region", first_row, last_row))

    def goto_position(self, row, column):
        self.calls.append(("goto_position", row, column))

    def set_background_color(self, index):
        self.calls.append(("set_background_color", index))

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


class AsyncKeyReaderTests(unittest.TestCase):
    def test_decodes_text_ascii_and_utf8(self):
        reader = terminals.AsyncKeyReader(fd=0)

        self.assertEqual(feed_bytes(reader, b"a"), [terminals.KeyEvent("TEXT", "a")])
        self.assertEqual(feed_bytes(reader, bytes([0xc3])), [])
        self.assertEqual(feed_bytes(reader, bytes([0xa9])), [terminals.KeyEvent("TEXT", "\u00e9")])

    def test_decodes_control_keys(self):
        reader = terminals.AsyncKeyReader(fd=0)

        events = feed_bytes(reader, b"\x03\x04\r\n\x7f\x08")

        self.assertEqual(
            events,
            [
                terminals.KeyEvent("CTRL_C"),
                terminals.KeyEvent("CTRL_D"),
                terminals.KeyEvent("ENTER"),
                terminals.KeyEvent("ENTER"),
                terminals.KeyEvent("BACKSPACE"),
                terminals.KeyEvent("BACKSPACE"),
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

    def test_bracketed_paste_keeps_newlines_as_text(self):
        reader = terminals.AsyncKeyReader(fd=0)

        events = feed_bytes(reader, b"\x1b[200~a\nb\x1b[201~")

        self.assertEqual(
            events,
            [
                terminals.KeyEvent("PASTE_START"),
                terminals.KeyEvent("TEXT", "a"),
                terminals.KeyEvent("TEXT", "\n"),
                terminals.KeyEvent("TEXT", "b"),
                terminals.KeyEvent("PASTE_END"),
            ],
        )
        self.assertFalse(reader.paste_mode)

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


class InputBufferTests(unittest.TestCase):
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
            def __init__(self, fd, watch_resize=False):
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
                ("set_clipping_region", 10, 13),
                ("goto_position", 1, 1),
                ("set_background_color", terminals.INPUT_COLOR),
                ("clear_to_end_of_screen",),
                ("update_status_bar",),
                ("set_clipping_region", 10, 13),
                ("goto_position", 1, 1),
                ("set_background_color", terminals.INPUT_COLOR),
                ("save_cursor_position",),
                ("restore_cursor_position",),
                ("flush",),
            ],
        )


class MenuTests(unittest.TestCase):
    def run_menu_with_inputs(self, inputs):
        old_get_input_async = terminals.get_input_async
        old_terminal = terminals.terminal
        answers = list(inputs)
        recorder = RecordingTerminal()

        async def fake_get_input_async(prompt=None, history=None):
            return answers.pop(0)

        try:
            terminals.get_input_async = fake_get_input_async
            terminals.terminal = recorder
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = asyncio.run(terminals.run_menu_async(["alpha", "beta"]))
            return result, out.getvalue(), recorder.calls
        finally:
            terminals.get_input_async = old_get_input_async
            terminals.terminal = old_terminal

    def test_menu_number_selection_allows_trailing_period(self):
        result, output, calls = self.run_menu_with_inputs(["2."])

        self.assertEqual(result, "beta")
        self.assertIn("1. alpha", output)
        self.assertIn("2. beta", output)
        self.assertEqual(calls.count(("save_cursor_position",)), 2)

    def test_menu_name_selection_returns_name(self):
        result, output, calls = self.run_menu_with_inputs(["custom-model"])

        self.assertEqual(result, "custom-model")
        self.assertIn("1. alpha", output)
        self.assertIn("2. beta", output)
        self.assertEqual(calls.count(("save_cursor_position",)), 2)


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
