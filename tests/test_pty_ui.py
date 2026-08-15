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


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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


def run_loki_pty_reply(stream: bool, stream_chunks=None):
    """Run one real TUI turn; optionally pause after its first delta.

    Returns ``(all_output, before_stream_release)``. The second value is only
    populated for a genuine dummy-provider delta stream, and is captured while
    the provider is still blocked before producing its remaining deltas.
    """
    tmpdir = tempfile.mkdtemp(prefix="loki-pty-test-")
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

        os.write(master, b"hi\r")
        reply_output = _read_with_timeout(master, 4.0)
        collected += reply_output
        if gate:
            before_stream_release = reply_output
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
