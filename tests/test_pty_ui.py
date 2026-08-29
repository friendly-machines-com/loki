"""End-to-end UI test: run the real loki_agent TUI under a pty with the
dummy provider (no network) and assert on what the terminal actually shows.

The raw-byte assertions work everywhere. When pyte is importable, the same
run is additionally decoded into a pyte screen so SGR attributes can be
checked as attributes (bold / cyan), not just as substrings.
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
import sys
import tempfile
import termios
import time
import unittest


try:
    import pyte
except ImportError:
    pyte = None

ROOT = pathlib.Path(__file__).resolve().parents[1]

REPLY = "**boldword** and `codeword` done"
BOLD_RUN = b"\x1b[1mboldword\x1b[0m"
CODE_RUN = b"\x1b[36mcodeword\x1b[0m"


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
                       initial_input=b"hi"):
    """Run one real TUI turn; optionally pause after its first delta.

    Returns ``(all_output, before_stream_release)``. The second value is only
    populated for a genuine dummy-provider delta stream, and is captured while
    the provider is still blocked before producing its remaining deltas.
    """
    tmpdir = tempfile.mkdtemp(prefix="loki-pty-test-")
    if create_image:
        pathlib.Path(tmpdir, "image.png").write_bytes(
            b"\x89PNG\r\n\x1a\npayload")
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
                    "".join(stream_chunks) if stream_chunks else REPLY),
                "LOKI_STREAM": "1" if stream else "0",
                "PYTHONPATH": str(ROOT),
            }
            if stream_chunks:
                env["LOKI_DUMMY_STREAM_CHUNKS"] = json.dumps(stream_chunks)
                env["LOKI_DUMMY_STREAM_GATE"] = gate
            os.execvpe(
                sys.executable, [sys.executable, "-m", "loki_agent"], env)
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

    @unittest.skipIf(pyte is None, "pyte not installed")
    def test_pyte_screen_attributes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                if stream:
                    chunks = [
                        "visible prefix ",
                        "**boldword** and `codeword` done",
                    ]
                else:
                    chunks = None
                output, _before_release = run_loki_pty_reply(
                    stream=stream, stream_chunks=chunks)
                screen = pyte.Screen(80, 24)
                pyte_stream = pyte.Stream(screen)
                pyte_stream.feed(output.decode("utf-8", errors="replace"))

                text = "\n".join(screen.display)
                row = next((i for i, line in enumerate(screen.display)
                            if "boldword" in line), None)
                self.assertIsNotNone(
                    row, f"reply not on screen; got:\n{text}")
                col = screen.display[row].index("boldword")
                self.assertTrue(screen.buffer[row][col].bold)
                self.assertEqual(screen.buffer[row][col].data, "b")

                row = next(i for i, line in enumerate(screen.display)
                           if "codeword" in line)
                col = screen.display[row].index("codeword")
                self.assertEqual(screen.buffer[row][col].fg, "cyan")
                self.assertEqual(screen.buffer[row][col].data, "c")


if __name__ == "__main__":
    unittest.main()


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
