"""End-to-end UI test: run the real loki_agent TUI under a pty with the
dummy provider (no network) and assert on what the terminal actually shows.

The assertions check the SGR runs (bold / cyan) as raw byte substrings of
what was written to the tty.
"""

import fcntl
import json
import os
import pathlib
import pty
import select
import shutil
import signal
import struct
import tempfile
import termios
import time
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
LOKI = ROOT / "loki.py"

REPLY = "**boldword** and `codeword` done"
BOLD_RUN = b"\x1b[1mboldword\x1b[0m"
CODE_RUN = b"\x1b[36mcodeword\x1b[0m"


class _SgrStreamTracker:
    """Tiny VT byte-stream tracker, stdlib only.

    Parses escape-sequence boundaries (CSI, OSC, DCS/SOS/PM/APC, 2-char
    and charset escapes) and records the SGR attribute state that was
    active while each printable character was emitted. It deliberately
    does NOT emulate a screen: no grid, no cursor addressing, no
    erase/scroll semantics. It validates the byte stream itself --
    sequences terminate, styles end reset -- and where attributes were
    active. It proves nothing about any real terminal.
    """

    ANSI_FG = {30: "black", 31: "red", 32: "green", 33: "yellow",
               34: "blue", 35: "magenta", 36: "cyan", 37: "white"}

    def __init__(self):
        self.bold = False
        self.fg = "default"
        self.text = []      # printable characters, in emission order
        self.states = []    # (bold, fg) per character
        self.unterminated = 0

    # -- SGR -----------------------------------------------------------

    def _apply_sgr(self, params: bytes):
        fields = params.decode("ascii", "replace").split(";")
        if fields == [""]:
            fields = ["0"]
        i = 0
        while i < len(fields):
            raw = fields[i]
            n = int(raw) if raw.isdigit() else 0
            if n == 0:
                self.bold = False
                self.fg = "default"
            elif n == 1:
                self.bold = True
            elif n == 22:
                self.bold = False
            elif n in self.ANSI_FG:
                self.fg = self.ANSI_FG[n]
            elif n == 39:
                self.fg = "default"
            elif n == 38:  # 38;5;n or 38;2;r;g;b extended color
                if i + 1 < len(fields) and fields[i + 1] == "5":
                    i += 2
                    self.fg = "indexed"
                elif i + 1 < len(fields) and fields[i + 1] == "2":
                    i += 4
                    self.fg = "rgb"
                else:
                    self.fg = "unknown"
            i += 1

    # -- byte-stream parsing --------------------------------------------

    def _escape(self, data: bytes, i: int) -> int:
        n = len(data)
        if i + 1 >= n:
            self.unterminated += 1
            return n
        c = data[i + 1]
        if c == 0x5B:  # ESC [ : CSI
            j = i + 2
            while j < n and 0x30 <= data[j] <= 0x3F:   # parameter bytes
                j += 1
            while j < n and 0x20 <= data[j] <= 0x2F:   # intermediates
                j += 1
            if j >= n:
                self.unterminated += 1
                return n
            if data[j] == 0x6D:  # final 'm'
                self._apply_sgr(data[i + 2:j])
            return j + 1
        if c in (0x5D, 0x50, 0x58, 0x5E, 0x5F):  # OSC/DCS/SOS/PM/APC
            j = i + 2
            while j < n:
                if data[j] == 0x07:                      # BEL terminator
                    return j + 1
                if data[j] == 0x1B and j + 1 < n and data[j + 1] == 0x5C:
                    return j + 2                          # ST terminator
                j += 1
            self.unterminated += 1
            return n
        j = i + 1
        while j < n and 0x20 <= data[j] <= 0x2F:  # charset/intermediates
            j += 1
        if j >= n:
            self.unterminated += 1
            return n
        return j + 1  # single final byte: ESC 7, ESC 8, ESC ( B, ...

    def feed(self, data: bytes):
        i, n = 0, len(data)
        while i < n:
            b = data[i]
            if b == 0x1B:
                i = self._escape(data, i)
            elif 0x20 <= b != 0x7F:
                if b < 0x80:
                    self.text.append(chr(b))
                    self.states.append((self.bold, self.fg))
                    i += 1
                else:
                    width = 2 if b < 0xE0 else 3 if b < 0xF0 else 4
                    try:
                        ch = data[i:i + width].decode("utf-8")
                    except UnicodeDecodeError:
                        i += 1
                        continue
                    self.text.append(ch)
                    self.states.append((self.bold, self.fg))
                    i += width
            else:
                i += 1  # C0 control bytes carry no SGR state

    # -- queries ----------------------------------------------------------

    def printed_with(self, needle: str, *, bold=None, fg=None) -> bool:
        """True if `needle` was emitted with the attribute(s) active."""
        hay = "".join(self.text)
        start = 0
        while True:
            k = hay.find(needle, start)
            if k < 0:
                return False
            span = self.states[k:k + len(needle)]
            ok = True
            if bold is not None:
                ok = ok and all(s[0] == bold for s in span)
            if fg is not None:
                ok = ok and all(s[1] == fg for s in span)
            if ok:
                return True
            start = k + 1


def _set_size(fd):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))


def _read_with_timeout(master, total=4.0):
    """Drain the pty; reset the deadline on each arriving chunk."""
    buf = b""
    deadline = time.time() + total
    while time.time() < deadline:
        remaining = max(0.05, deadline - time.time())
        r, _, _ = select.select([master], [], [], min(0.2, remaining))
        if not r:
            if buf:
                break  # quiesced
            continue
        try:
            chunk = os.read(master, 65536)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        deadline = time.time() + 0.5
    return buf


def run_loki_pty_reply(stream: bool, stream_chunks=None,
                       queued_inputs=None, create_image=False,
                       initial_input=b"hi", reply=None,
                       raw_file_data=None):
    """Run one real TUI turn; optionally pause after its first delta.

    Returns ``(all_output, before_stream_release)``. The second value is only
    populated for a genuine dummy-provider delta stream, and is captured while
    the provider is still blocked before producing its remaining deltas.
    """
    tmpdir = tempfile.mkdtemp(prefix="loki-pty-test-")
    if create_image:
        pathlib.Path(tmpdir, "image.png").write_bytes(
            b"\x89PNG\r\n\x1a\npayload")
    if raw_file_data is not None:
        pathlib.Path(tmpdir, "attack.bin").write_bytes(raw_file_data)
    gate = os.path.join(tmpdir, "release-stream") if stream_chunks else None
    pid, master = pty.fork()
    if pid == 0:  # child: real tty on stdin/stdout, hermetic dirs
        try:
            _set_size(0)
            os.chdir(tmpdir)
            env = {
                "HOME": tmpdir,
                "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
                "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
                "PATH": os.environ.get("PATH", ""),
                "TERM": "xterm",
                "LOKI_PROVIDER": "dummy",
                "LOKI_API_BASE": "http://dummy.invalid/v1",
                "LOKI_DUMMY_REPLY": (
                    "".join(stream_chunks) if stream_chunks
                    else REPLY if reply is None else reply),
                "LOKI_STREAM": "1" if stream else "0",
            }
            if stream_chunks:
                env["LOKI_DUMMY_STREAM_CHUNKS"] = json.dumps(stream_chunks)
                env["LOKI_DUMMY_STREAM_GATE"] = gate
            os.execve(str(LOKI), [str(LOKI)], env)
        except Exception:
            pass
        os._exit(127)

    collected = b""
    before_stream_release = b""
    try:
        _set_size(master)
        collected += _read_with_timeout(master, 6.0)  # startup banner

        os.write(master, initial_input + b"\r")
        reply_output = _read_with_timeout(master, 4.0)
        collected += reply_output
        if gate:
            before_stream_release = reply_output
            for queued_input in queued_inputs or []:
                os.write(master, queued_input.encode() + b"\r")
                queued_output = _read_with_timeout(master, 2.0)
                collected += queued_output
                before_stream_release += queued_output
            pathlib.Path(gate).touch()
            collected += _read_with_timeout(master, 4.0)

        os.write(master, b"/quit\r")
        collected += _read_with_timeout(master, 2.0)
        for _ in range(40):
            try:
                done_pid, status = os.waitpid(pid, os.WNOHANG)
            except OSError:
                break
            if done_pid:
                break
            time.sleep(0.1)
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except OSError:
                break
            time.sleep(0.2)
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass
        os.close(master)
        shutil.rmtree(tmpdir, ignore_errors=True)
    return collected, before_stream_release


@unittest.skipUnless(hasattr(os, "fork"), "needs fork/pty")
class PtyUiTests(unittest.TestCase):

    def _assert_styled_output(self, output):
        # The reply must be styled, i.e. the SGR runs around the words are
        # actually written to the tty. This is the assertion the old
        # streaming path could not pass: it printed raw markdown.
        self.assertIn(BOLD_RUN, output)
        self.assertIn(CODE_RUN, output)

    def test_batch_reply_is_styled_on_tty(self):
        output, _before_release = run_loki_pty_reply(stream=False)
        self._assert_styled_output(output)

    def test_batch_and_split_stream_model_controls_are_not_executed(self):
        attack = "\x1b]777;LOKI_MODEL_ATTACK\x07"
        visible = b"^[]777;LOKI_MODEL_ATTACK^G"

        batch, _ = run_loki_pty_reply(
            stream=False, reply=f"before {attack} **boldword**")
        streamed, _ = run_loki_pty_reply(
            stream=True,
            stream_chunks=[
                "before \x1b]",
                "777;LOKI_MODEL_ATTACK",
                "\x07 **bold",
                "word**",
            ],
        )

        for output in [batch, streamed]:
            self.assertNotIn(attack.encode(), output)
            self.assertIn(visible, output)
            self.assertIn(BOLD_RUN, output)

    def test_pasted_terminal_controls_are_displayed_not_executed(self):
        attack = b"\x1b]777;LOKI_INPUT_ATTACK\x07"
        pasted = (
            b"\x1b[200~"
            b"first" + attack + b"\t\nnext"
            b"\x1b[201~"
        )

        output, _before_release = run_loki_pty_reply(
            stream=False, initial_input=pasted)

        self.assertNotIn(attack, output)
        self.assertIn(
            b"first^[]777;LOKI_INPUT_ATTACK^G^I\r\nnext",
            output,
        )

    def test_explicit_bang_command_keeps_raw_unix_output(self):
        attack = b"\x1b]777;LOKI_COMMAND_OUTPUT\x07"

        output, _ = run_loki_pty_reply(
            stream=False,
            initial_input=b"!cat attack.bin",
            raw_file_data=attack,
        )

        self.assertIn(attack, output)

    def test_status_bar_shows_api_and_mode(self):
        # Regression for the frontend split dropping the status text
        # registration: the bar must carry text, not just background color.
        output, _before_release = run_loki_pty_reply(stream=False)
        self.assertIn(b"Remote: API: dummy.invalid", output)
        self.assertIn(
            b"Local: turn: idle, queued messages: 0, "
            b"queued images: 0, mode: normal",
            output,
        )

    def test_streamed_plain_prefix_is_visible_before_completion(self):
        chunks = [
            "visible before completion",
            " and **boldword** plus `codeword` done",
        ]
        output, before_release = run_loki_pty_reply(
            stream=True, stream_chunks=chunks)

        self.assertIn(b"visible before completion", before_release)
        self.assertNotIn(b"boldword", before_release)
        self.assertNotIn(b"LLM Response Time", before_release)
        self._assert_styled_output(output)

    def test_status_bar_tracks_messages_queued_behind_active_turn(self):
        output, before_release = run_loki_pty_reply(
            stream=True,
            stream_chunks=["blocked prefix", " completed"],
            queued_inputs=["second", "third"],
        )

        self.assertIn(b"queued messages: 1, queued images: 0", before_release)
        self.assertIn(b"queued messages: 2, queued images: 0", before_release)
        self.assertIn(
            b"turn: running, queued messages: 2, queued images: 0",
            before_release,
        )
        self.assertIn(
            b"turn: idle, queued messages: 0, queued images: 0",
            output,
        )
        self.assertIn(b"queued messages: 0, queued images: 0", output)

    def test_status_bar_tracks_image_after_queued_command_is_validated(self):
        output, before_release = run_loki_pty_reply(
            stream=True,
            stream_chunks=["blocked prefix", " completed"],
            queued_inputs=["/image image.png"],
            create_image=True,
        )

        self.assertIn(b"queued messages: 1, queued images: 0", before_release)
        self.assertIn(b"queued messages: 0, queued images: 1", output)

    def test_full_stream_parses_and_styles_land(self):
        # Whole-stream validation of the tty byte output: every escape
        # sequence terminates, the stream ends with SGR state reset (no
        # dangling styles), and the styled words were emitted with their
        # attributes actually active. This is a spec-shaped parser check
        # of Loki's output -- it does not emulate a screen and proves
        # nothing about any real terminal.
        for stream in (False, True):
            with self.subTest(stream=stream):
                chunks = (["visible prefix ",
                           "**boldword** and `codeword` done"]
                          if stream else None)
                output, _before_release = run_loki_pty_reply(
                    stream=stream, stream_chunks=chunks)

                tracker = _SgrStreamTracker()
                tracker.feed(output)

                self.assertEqual(
                    tracker.unterminated, 0,
                    "escape sequence ran past end of stream")
                self.assertFalse(
                    tracker.bold, "stream ended with bold still active")
                self.assertEqual(
                    tracker.fg, "default",
                    "stream ended with a foreground color still active")
                self.assertTrue(
                    tracker.printed_with("boldword", bold=True),
                    "boldword was never emitted with bold active")
                self.assertTrue(
                    tracker.printed_with("codeword", fg="cyan"),
                    "codeword was never emitted with cyan foreground")


@unittest.skipUnless(hasattr(os, "fork"), "needs fork/pty")
class PtyCliUsageTests(unittest.TestCase):
    """--help and argument errors must not touch the terminal.

    These exits run before initialize_terminal_overlay. With stdin AND
    stdout on a real tty (the real Terminal class is selected at import
    time), they must emit no escape sequences at all -- otherwise every
    usage error enters and leaves TUI mode, hiding the cursor, setting
    scroll regions, and clearing the user's screen on the way out.
    """

    def _run_cli(self, cwd, *cli_args):
        pid, master = pty.fork()
        if pid == 0:  # child: pty on stdin/stdout, hermetic dirs
            try:
                _set_size(0)
                os.chdir(cwd)
                env = {
                    "HOME": cwd,
                    "XDG_CONFIG_HOME": os.path.join(cwd, "config"),
                    "XDG_STATE_HOME": os.path.join(cwd, "state"),
                    "PATH": os.environ.get("PATH", ""),
                    "TERM": "xterm",
                }
                os.execve(
                    str(LOKI), [str(LOKI), *cli_args], env)
            except Exception:
                pass
            os._exit(127)
        output = b""
        exit_code = None
        try:
            _set_size(master)
            output = _read_with_timeout(master, 4.0)
            status = None
            # Bounded wait: a usage-path regression that waits for input
            # instead of exiting must FAIL the test, not hang the suite.
            for _ in range(50):  # up to 5s
                try:
                    done_pid, status = os.waitpid(pid, os.WNOHANG)
                except OSError:
                    break
                if done_pid:
                    break
                time.sleep(0.1)
            if status is not None:
                exit_code = os.waitstatus_to_exitcode(status)
        finally:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except OSError:
                    break
                time.sleep(0.2)
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
            os.close(master)
        if exit_code is None:
            self.fail(
                f"loki{tuple(cli_args)!r} never exited on the usage path; "
                f"output captured: {output!r}")
        return exit_code, output

    def test_help_and_arg_errors_leave_terminal_untouched(self):
        # Invariants only: the expected exit code, help actually printing
        # something, a bad option being named back to the user, and not a
        # single escape byte -- usage exits must not enter/leave TUI mode.
        cases = [
            (("--help",), 0),
            (("--definitely-not-an-option",), 2),
        ]
        for cli_args, expected_exit in cases:
            with self.subTest(args=cli_args):
                with tempfile.TemporaryDirectory() as cwd:
                    exit_code, output = self._run_cli(cwd, *cli_args)
                self.assertEqual(exit_code, expected_exit)
                self.assertNotIn(
                    b"\x1b", output,
                    "usage exits must not emit escape sequences; got: "
                    f"{output!r}")
                if expected_exit == 0:
                    self.assertTrue(output, "help printed nothing")
                else:
                    self.assertIn(
                        cli_args[0].encode(), output,
                        "the rejected option must be named back to the "
                        "user")

    def test_argument_error_represents_supplied_terminal_controls(self):
        attack = "\x1b]777;LOKI_OPTION_ATTACK\x07"
        with tempfile.TemporaryDirectory() as cwd:
            exit_code, output = self._run_cli(
                cwd, "--not-an-option-" + attack)

        self.assertEqual(exit_code, 2)
        self.assertNotIn(attack.encode(), output)
        self.assertNotIn(b"\x1b", output)
        self.assertIn(b"\\x1b", output)


@unittest.skipUnless(hasattr(os, "fork"), "needs fork/pty")
class PtyCtrlCTests(unittest.TestCase):
    """Ctrl+C must reach the reader as byte 0x03, not become SIGINT.

    With ISIG left set on the tty, the driver eats Ctrl+C and delivers
    SIGINT to the foreground process group before the byte reaches
    AsyncKeyReader, so cancel_requested never becomes True and a running
    turn cannot be cancelled between tool calls.  These tests pin the
    ISIG-clearing behavior of TerminalMode end to end.
    """

    def test_terminal_mode_clears_isig(self):
        # Direct: enter TerminalMode on a fresh pty and inspect lflag bits.
        pid, master = pty.fork()
        if pid == 0:
            # Child: never returns from this block.
            try:
                import sys as _sys
                import termios as _termios
                import time as _time
                from loki_agent import terminals as _terminals

                mode = _terminals.TerminalMode(0, enabled=True)
                mode.__enter__()
                attrs = _termios.tcgetattr(0)
                _sys.stdout.buffer.write(
                    b"ISIG_SET\n" if attrs[3] & _termios.ISIG
                    else b"ISIG_CLEAR\n")
                _sys.stdout.buffer.flush()
                mode.__exit__(None, None, None)
                attrs = _termios.tcgetattr(0)
                _sys.stdout.buffer.write(
                    b"RESTORED_ISIG_SET\n" if attrs[3] & _termios.ISIG
                    else b"RESTORED_ISIG_CLEAR\n")
                _sys.stdout.buffer.flush()
                _time.sleep(0.5)
            except Exception:
                pass
            os._exit(0)
        try:
            out = _read_with_timeout(master, 4.0)
        finally:
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
            os.close(master)
        self.assertIn(b"ISIG_CLEAR", out)
        self.assertNotIn(b"ISIG_SET\n", out.replace(b"ISIG_CLEAR", b""))
        self.assertIn(b"RESTORED_ISIG_SET", out)

    def test_ctrl_c_sets_cancel_flag_in_reader(self):
        # End to end: with ISIG clear, a 0x03 byte written to the pty is
        # seen by AsyncKeyReader as CTRL_C and sets cancel_requested.
        pid, master = pty.fork()
        if pid == 0:
            try:
                import sys as _sys
                import asyncio as _asyncio
                import time as _time
                from loki_agent import terminals as _terminals

                async def _main():
                    mode = _terminals.TerminalMode(0, enabled=True)
                    mode.__enter__()
                    reader = _terminals.AsyncKeyReader(0)
                    key_kind = None
                    async with reader:
                        # Drive the reader the way the real loop does; the
                        # flag is set inside read_key's parse, not by the
                        # byte arriving alone.
                        try:
                            key = await _asyncio.wait_for(
                                reader.read_key(), timeout=3.0)
                            key_kind = key.kind
                        except _asyncio.TimeoutError:
                            pass
                    mode.__exit__(None, None, None)
                    if reader.cancel_requested and key_kind == "CTRL_C":
                        _sys.stdout.buffer.write(b"CANCEL_SET\n")
                    else:
                        _sys.stdout.buffer.write(
                            f"CANCEL_NOT_SET key={key_kind}\n".encode())
                    _sys.stdout.buffer.write(
                        b"CANCEL_SET\n" if reader.cancel_requested
                        else b"CANCEL_NOT_SET\n")
                    _sys.stdout.buffer.flush()
                    _time.sleep(0.3)
                _asyncio.run(_main())
            except Exception:
                pass
            os._exit(0)
        try:
            _set_size(master)
            _read_with_timeout(master, 1.0)  # let child reach its wait
            os.write(master, b"\x03")
            out = _read_with_timeout(master, 4.0)
        finally:
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
            os.close(master)
        self.assertIn(b"CANCEL_SET", out)
        self.assertNotIn(b"CANCEL_NOT_SET", out)


@unittest.skipUnless(hasattr(os, "fork"), "needs fork/pty")
class PtyTurnCancelTests(unittest.TestCase):
    """A real turn: Ctrl+C must reach the reader's cancel event.

    The reader's cancel_event is what run_foreground races against to
    interrupt a foreground job; this pins the wiring from tty byte to
    event with the real InputSession machinery (per-turn clear included).
    """

    def test_ctrl_c_sets_reader_event_after_read_key(self):
        pid, master = pty.fork()
        if pid == 0:
            try:
                import sys as _sys
                import asyncio as _asyncio
                import time as _time
                from loki_agent import terminals as _terminals

                async def _main():
                    mode = _terminals.TerminalMode(0, enabled=True)
                    mode.__enter__()
                    reader = _terminals.AsyncKeyReader(0)
                    key_kind = None
                    async with reader:
                        reader.cancel_event.clear()
                        try:
                            key = await _asyncio.wait_for(
                                reader.read_key(), timeout=3.0)
                            key_kind = key.kind
                        except _asyncio.TimeoutError:
                            pass
                    mode.__exit__(None, None, None)
                    if (key_kind == "CTRL_C"
                            and reader.cancel_event.is_set()):
                        _sys.stdout.buffer.write(b"EVENT_SET\n")
                    else:
                        _sys.stdout.buffer.write(b"EVENT_NOT_SET\n")
                    _sys.stdout.buffer.flush()
                    _time.sleep(0.2)
                _asyncio.run(_main())
            except Exception:
                pass
            os._exit(0)
        try:
            _set_size(master)
            _read_with_timeout(master, 1.0)
            os.write(master, b"\x03")
            out = _read_with_timeout(master, 4.0)
        finally:
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
            os.close(master)
        self.assertIn(b"EVENT_SET", out)
        self.assertNotIn(b"EVENT_NOT_SET", out)


if __name__ == "__main__":
    unittest.main()
