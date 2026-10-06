"""End-to-end UI test: run the real loki_agent TUI under a pty with the
dummy provider (no network) and assert on what the terminal actually shows.

The assertions parse the byte stream and check what the escapes *mean* --
which text was emitted bold, which was emitted cyan, whether any OSC-777
control sequence was emitted -- rather than matching exact escape bytes,
because a ConPTY pipe rewrites SGR spellings while preserving their meaning.
"""

import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The ``--pty-child`` process runs this file as a script with the tests
# directory as ``sys.path[0]``; make the package importable there too.
sys.path.insert(0, str(ROOT))

from loki_agent import pty_backend  # noqa: E402
from loki_entrypoints import (child_environment, configure_container,  # noqa: E402
                              entrypoint)

REPLY = "**boldword** and `codeword` done"


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
        self.osc_payloads = []  # decoded OSC payloads, in emission order

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
        if c in [0x5D, 0x50, 0x58, 0x5E, 0x5F]:  # OSC/DCS/SOS/PM/APC
            j = i + 2
            while j < n:
                if data[j] == 0x07:                      # BEL terminator
                    end = j + 1
                    break
                if data[j] == 0x1B and j + 1 < n and data[j + 1] == 0x5C:
                    end = j + 2                          # ST terminator
                    break
                j += 1
            else:
                self.unterminated += 1
                return n
            if c == 0x5D:  # OSC: keep the payload for the control-sequence check
                self.osc_payloads.append(
                    data[i + 2:j].decode("ascii", "replace"))
            return end
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

    def osc_param_emitted(self, first_param: str) -> bool:
        """True if any recorded OSC payload's first parameter is `first_param`.

        An OSC payload is the bytes between the ``ESC ]`` introducer and its
        terminator; the first parameter is everything before the first ``;``
        (or the whole payload when there is no separator).
        """
        for payload in self.osc_payloads:
            if payload.split(";", 1)[0] == first_param:
                return True
        return False


def _read_until(handle, output, expected, *, start=0):
    """Wait for newly emitted text, retaining the unmodified terminal stream."""
    previous = _SgrStreamTracker()
    previous.feed(bytes(output[:start]))
    text_start = len(previous.text)
    while True:
        # Reparse the complete capture: a read can split any VT sequence or
        # UTF-8 character, and this tracker is not an incremental parser.
        tracker = _SgrStreamTracker()
        tracker.feed(bytes(output))
        if expected in "".join(tracker.text[text_start:]):
            return
        chunk = handle.read(65536)
        if not chunk:
            raise AssertionError(
                f"PTY exited with {handle.wait()} before emitting {expected!r}; "
                f"output: {bytes(output)!r}")
        output.extend(chunk)


def _read_to_exit(handle, output=None):
    """Drain through terminal EOF before waiting for the process."""
    if output is None:
        output = bytearray()
    while True:
        chunk = handle.read(65536)
        if not chunk:
            return handle.wait(), bytes(output)
        output.extend(chunk)


def run_loki_pty_reply(stream: bool, stream_chunks=None,
                       queued_inputs=None, create_image=False,
                       initial_input=b"hi", reply=None,
                       raw_file_data=None, stream_prefix=None):
    """Run one real TUI turn; optionally pause after its first delta.

    Returns ``(all_output, before_stream_release)``. The second value is only
    populated for a genuine dummy-provider delta stream, and is captured while
    the provider is still blocked before producing its remaining deltas.
    """
    if stream_chunks and (not stream or not stream_prefix):
        raise ValueError('a gated stream requires its expected visible prefix')
    root = tempfile.mkdtemp(prefix="loki-pty-test-")
    tmpdir = os.path.join(root, "workspace")
    os.mkdir(tmpdir)
    if create_image:
        pathlib.Path(tmpdir, "image.png").write_bytes(
            b"\x89PNG\r\n\x1a\npayload")
    if raw_file_data is not None:
        pathlib.Path(tmpdir, "attack.bin").write_bytes(raw_file_data)
    gate = os.path.join(tmpdir, "release-stream") if stream_chunks else None
    env = {
        "HOME": root,
        "XDG_CONFIG_HOME": os.path.join(root, "config"),
        "XDG_STATE_HOME": os.path.join(root, "state"),
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
    env = child_environment(**env)
    handle = None
    collected = bytearray()
    before_stream_release = b""
    try:
        configure_container(env, tmpdir)
        handle = pty_backend.spawn_pty(
            [entrypoint("loki")], env=env, cwd=tmpdir)
        _read_until(handle, collected, 'User: ')
        turn_start = len(collected)
        handle.write(initial_input + b"\r")
        if gate:
            _read_until(handle, collected, stream_prefix)
            queued_texts = 0
            for queued_input in queued_inputs or []:
                handle.write(queued_input.encode() + b"\r")
                if queued_input == '/ps':
                    _read_until(handle, collected,
                                'No running, starting, or failed jobs.')
                else:
                    queued_texts += 1
                    _read_until(
                        handle, collected,
                        'turn: running, mode: normal; '
                        f'/queue(texts: {queued_texts}, images: 0)')
            before_stream_release = bytes(collected[turn_start:])
            release_start = len(collected)
            pathlib.Path(gate).touch()
            # With no queued inputs, wait for the turn to end before causing
            # a prompt redraw: split model controls must remain observable as
            # contiguous escaped text. Queued-input tests deliberately permit
            # concurrent redraws and use the FIFO completion below.
            if not queued_inputs:
                _read_until(
                    handle, collected,
                    'turn: idle, mode: normal; /queue(texts: 0, images: 0)',
                    start=release_start)

        # /quit uses the same FIFO as prompts and /image. It cannot overtake
        # them, so natural exit is our completion acknowledgment.
        handle.write(b"/quit\r")
        exit_code, output = _read_to_exit(handle, collected)
        if exit_code != 0:
            raise AssertionError(f'loki exited with {exit_code}; output: {output!r}')
    finally:
        try:
            if handle is not None:
                try:
                    handle.terminate()
                finally:
                    handle.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return output, before_stream_release


class PtyCaptureTests(unittest.TestCase):
    def test_marker_survives_split_controls_and_utf8(self):
        handle = mock.Mock(spec=pty_backend.PtyHandle)
        handle.read.side_effect = [b'\x1b[3', b'2mREA', b'DY \xc3', b'\xa9\x1b[0m']
        output = bytearray()
        _read_until(handle, output, 'READY \u00e9')
        self.assertEqual(bytes(output), b'\x1b[32mREADY \xc3\xa9\x1b[0m')
        handle.wait.assert_not_called()

    def test_capture_keeps_trailing_controls_and_nonzero_exit(self):
        handle = mock.Mock(spec=pty_backend.PtyHandle)
        handle.read.side_effect = [b'first', b'last\x1b[0m', b'']
        handle.wait.return_value = 2
        self.assertEqual(_read_to_exit(handle), (2, b'firstlast\x1b[0m'))
        handle.wait.assert_called_once_with()

    def test_exit_before_marker_reports_output_and_status(self):
        handle = mock.Mock(spec=pty_backend.PtyHandle)
        handle.read.side_effect = [b'failed startup', b'']
        handle.wait.return_value = 2
        with self.assertRaisesRegex(AssertionError, '2.*READY.*failed startup'):
            _read_until(handle, bytearray(), 'READY')

    def test_new_observation_does_not_reuse_an_old_marker(self):
        handle = mock.Mock(spec=pty_backend.PtyHandle)
        handle.read.side_effect = [b'REA', b'DY']
        output = bytearray(b'READY')
        _read_until(handle, output, 'READY', start=len(output))
        self.assertEqual(bytes(output), b'READYREADY')
        self.assertEqual(handle.read.call_count, 2)

    @unittest.skipUnless(os.name == 'posix', 'POSIX preserves every emitted byte')
    def test_real_capture_drains_output_beyond_one_read_and_preserves_exit(self):
        handle = pty_backend.spawn_pty([
            sys.executable, str(pathlib.Path(__file__).resolve()), '--pty-child', 'capture'])
        try:
            output = bytearray()
            _read_until(handle, output, 'CAPTURE_READY')
            handle.write(b'x')
            exit_code, captured = _read_to_exit(handle, output)
            self.assertEqual(exit_code, 2)
            self.assertIn(b'z' * 131072 + b'\x1b[0mCAPTURE_END', captured)
            self.assertEqual(handle.poll(), 2)
            self.assertEqual(handle.wait(), 2)
        finally:
            try:
                handle.terminate()
            finally:
                handle.close()

    def test_read_failure_is_not_eof(self):
        handle = mock.Mock(spec=pty_backend.PtyHandle)
        handle.read.side_effect = OSError('read failed')
        with self.assertRaisesRegex(OSError, 'read failed'):
            _read_to_exit(handle)
        handle.wait.assert_not_called()


class PtyFixtureTests(unittest.TestCase):
    def test_private_trees_are_outside_the_granted_workspace(self):
        module = sys.modules[__name__]
        for cli in [False, True]:
            with self.subTest(cli=cli):
                configured = []

                def configure(env, workspace):
                    self.assertTrue(os.path.isdir(workspace))
                    for key in ['XDG_CONFIG_HOME', 'XDG_STATE_HOME']:
                        self.assertNotEqual(os.path.commonpath([workspace, env[key]]),
                                            workspace)
                    configured.append(workspace)

                def spawn(arguments, *, env, cwd):
                    self.assertEqual(configured, [cwd])
                    if not cli:
                        self.assertTrue(pathlib.Path(cwd, 'image.png').is_file())
                        self.assertTrue(pathlib.Path(cwd, 'attack.bin').is_file())
                    return mock.Mock(poll=mock.Mock(return_value=0))

                with (
                    mock.patch.object(module, 'configure_container', side_effect=configure),
                    mock.patch.object(module, 'entrypoint', return_value='/installed/loki'),
                    mock.patch.object(module, '_read_until'),
                    mock.patch.object(module, '_read_to_exit', return_value=(0, b'')),
                    mock.patch.object(pty_backend, 'spawn_pty', side_effect=spawn),
                ):
                    if cli:
                        with tempfile.TemporaryDirectory() as root:
                            PtyCliUsageTests()._run_cli(root, '--help')
                    else:
                        run_loki_pty_reply(False, create_image=True, raw_file_data=b'test')
                self.assertFalse(os.path.exists(configured[0]))


class PtyUiTests(unittest.TestCase):

    def _assert_styled_output(self, output):
        # The reply must be styled, i.e. the SGR runs around the words are
        # actually written to the tty. This is the assertion the old
        # streaming path could not pass: it printed raw markdown. Asserted
        # semantically (was this text emitted with bold / cyan active?)
        # rather than as exact escape bytes, because a ConPTY pipe rewrites
        # SGR spellings while preserving their meaning.
        tracker = _SgrStreamTracker()
        tracker.feed(output)
        self.assertTrue(
            tracker.printed_with("boldword", bold=True),
            "boldword was never emitted with bold active")
        self.assertTrue(
            tracker.printed_with("codeword", fg="cyan"),
            "codeword was never emitted with cyan foreground")

    def test_batch_reply_is_styled_on_tty(self):
        output, _before_release = run_loki_pty_reply(stream=False)
        self._assert_styled_output(output)

    def test_batch_and_split_stream_model_controls_are_not_executed(self):
        attack = "\x1b]777;LOKI_MODEL_ATTACK\x07"
        visible = b"^[]777;LOKI_MODEL_ATTACK^G"

        batch, _ = run_loki_pty_reply(
            stream=False, reply=f"before {attack} **boldword**")
        streamed, _ = run_loki_pty_reply(
            stream=True, stream_prefix="before ",
            stream_chunks=[
                "before \x1b]",
                "777;LOKI_MODEL_ATTACK",
                "\x07 **bold",
                "word**",
            ],
        )

        for output in [batch, streamed]:
            tracker = _SgrStreamTracker()
            tracker.feed(output)
            self.assertIn(visible, output)
            self.assertFalse(
                tracker.osc_param_emitted("777"),
                "the OSC-777 model control must be escaped, not executed")
            self.assertTrue(
                tracker.printed_with("boldword", bold=True),
                "boldword was never emitted with bold active")

    def test_pasted_terminal_controls_are_displayed_not_executed(self):
        attack = b"\x1b]777;LOKI_INPUT_ATTACK\x07"
        pasted = (
            b"\x1b[200~"
            b"first" + attack + b"\t\nnext"
            b"\x1b[201~"
        )

        output, _before_release = run_loki_pty_reply(
            stream=False, initial_input=pasted)

        tracker = _SgrStreamTracker()
        tracker.feed(output)
        self.assertIn(
            b"first^[]777;LOKI_INPUT_ATTACK^G^I\r\nnext",
            output,
        )
        self.assertFalse(
            tracker.osc_param_emitted("777"),
            "the pasted OSC-777 control must be displayed, not executed")

    def test_explicit_bang_command_output_is_terminal_safe(self):
        attack = b"\x1b]777;LOKI_COMMAND_OUTPUT\x07"

        output, _ = run_loki_pty_reply(
            stream=False,
            initial_input=b"!cat attack.bin",
            raw_file_data=attack,
        )

        tracker = _SgrStreamTracker()
        tracker.feed(output)
        self.assertIn(b"^[]777;LOKI_COMMAND_OUTPUT^G", output)
        self.assertFalse(
            tracker.osc_param_emitted("777"),
            "the bang-command OSC-777 output must be escaped, not executed")

    def test_status_bar_shows_api_and_mode(self):
        # Regression for the frontend split dropping the status text
        # registration: the bar must carry text, not just background color.
        output, _before_release = run_loki_pty_reply(stream=False)
        self.assertIn(b"Remote: API: dummy.invalid", output)
        self.assertRegex(
            output,
            rb"Local: CWD: [^\r\n]*, turn: idle, mode: normal; "
            rb"/queue\(texts: 0, images: 0\), /pwd",
        )

    def test_streamed_plain_prefix_is_visible_before_completion(self):
        chunks = [
            "visible before completion",
            " and **boldword** plus `codeword` done",
        ]
        output, before_release = run_loki_pty_reply(
            stream=True, stream_chunks=chunks, stream_prefix=chunks[0])

        self.assertIn(b"visible before completion", before_release)
        self.assertNotIn(b"boldword", before_release)
        self.assertNotIn(b"LLM Response Time", before_release)
        self._assert_styled_output(output)

    def test_status_bar_tracks_messages_queued_behind_active_turn(self):
        output, before_release = run_loki_pty_reply(
            stream=True,
            stream_chunks=["blocked prefix", " completed"],
            stream_prefix="blocked prefix",
            queued_inputs=["second", "third"],
        )

        # These assertions concern queue transitions, independently of styling.
        before_release = re.sub(rb"\x1b\[[0-9;]*m", b"", before_release)
        output = re.sub(rb"\x1b\[[0-9;]*m", b"", output)
        self.assertIn(b"/queue(texts: 1, images: 0)", before_release)
        self.assertIn(b"/queue(texts: 2, images: 0)", before_release)
        self.assertIn(
            b"turn: running, mode: normal; /queue(texts: 2, images: 0)",
            before_release,
        )
        self.assertIn(
            b"turn: idle, mode: normal; /queue(texts: 0, images: 0)",
            output,
        )
        self.assertIn(b"/queue(texts: 0, images: 0)", output)

    def test_ps_result_is_visible_during_stream_without_joining_prompt_queue(self):
        output, before_release = run_loki_pty_reply(
            stream=True, stream_chunks=["partial **bold", "word** and `codeword` done"],
            stream_prefix="partial ", queued_inputs=["second", "/ps"])
        # An echoed /ps is not proof of execution: its actual local result
        # must appear while the real frontend's provider is still blocked.
        self.assertIn(b"No running, starting, or failed jobs.", before_release)
        self.assertNotIn(b"codeword", before_release)
        plain = re.sub(rb"\x1b\[[0-9;]*m", b"", before_release)
        self.assertIn(b"/queue(texts: 1, images: 0)", plain)
        self.assertNotIn(b"/queue(texts: 2", plain)
        self._assert_styled_output(output)
        tracker = _SgrStreamTracker()
        tracker.feed(output)
        self.assertEqual(tracker.unterminated, 0)
        self.assertFalse(tracker.bold)
        self.assertEqual(tracker.fg, "default")

    def test_status_bar_tracks_image_after_queued_command_is_validated(self):
        output, before_release = run_loki_pty_reply(
            stream=True,
            stream_chunks=["blocked prefix", " completed"],
            stream_prefix="blocked prefix",
            queued_inputs=["/image image.png"],
            create_image=True,
        )

        # These assertions concern queue transitions, independently of styling.
        before_release = re.sub(rb"\x1b\[[0-9;]*m", b"", before_release)
        output = re.sub(rb"\x1b\[[0-9;]*m", b"", output)
        self.assertIn(b"/queue(texts: 1, images: 0)", before_release)
        self.assertIn(b"/queue(texts: 0, images: 1)", output)

    def test_full_stream_parses_and_styles_land(self):
        # Whole-stream validation of the tty byte output: every escape
        # sequence terminates, the stream ends with SGR state reset (no
        # dangling styles), and the styled words were emitted with their
        # attributes actually active. This is a spec-shaped parser check
        # of Loki's output -- it does not emulate a screen and proves
        # nothing about any real terminal.
        for stream in [False, True]:
            with self.subTest(stream=stream):
                chunks = (["visible prefix ",
                           "**boldword** and `codeword` done"]
                          if stream else None)
                output, _before_release = run_loki_pty_reply(
                    stream=stream, stream_chunks=chunks,
                    stream_prefix=chunks[0] if chunks else None)

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


class PtyCliUsageTests(unittest.TestCase):
    """--help and argument errors must not touch the terminal.

    These exits run before initialize_terminal_overlay. With stdin AND
    stdout on a real tty (the real Terminal class is selected at import
    time), they must emit no escape sequences at all -- otherwise every
    usage error enters and leaves TUI mode, hiding the cursor, setting
    scroll regions, and clearing the user's screen on the way out.
    """

    def _run_cli(self, cwd, *cli_args):
        env = child_environment(
            HOME=cwd,
            XDG_CONFIG_HOME=os.path.join(cwd, "config"),
            XDG_STATE_HOME=os.path.join(cwd, "state"),
            PATH=os.environ.get("PATH", ""),
            TERM="xterm",
        )
        workspace = os.path.join(cwd, "workspace")
        os.makedirs(workspace, exist_ok=True)
        configure_container(env, workspace)
        handle = pty_backend.spawn_pty(
            [entrypoint("loki"), *cli_args], env=env, cwd=workspace)
        try:
            return _read_to_exit(handle)
        finally:
            try:
                handle.terminate()
            finally:
                handle.close()

    def _assert_usage_exits_before_overlay(self, cli_args, expected_exit):
        # The usage path must return before initialize_terminal_overlay, so it
        # cannot have entered TUI mode.  In-process with a mock, so it runs on
        # every platform; the byte-level escape check is POSIX-only because the
        # pty is transparent there, while ConPTY's pipe carries conhost's own
        # frame regardless of what Loki wrote.
        import asyncio
        from unittest import mock

        from loki_agent import terminal_frontend

        with mock.patch.object(
                terminal_frontend, "initialize_terminal_overlay") as overlay:
            exit_code = asyncio.run(
                terminal_frontend._run_frontend(list(cli_args)))
        self.assertEqual(exit_code, expected_exit)
        overlay.assert_not_called()

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
                self._assert_usage_exits_before_overlay(cli_args, expected_exit)
                with tempfile.TemporaryDirectory() as cwd:
                    exit_code, output = self._run_cli(cwd, *cli_args)
                self.assertEqual(exit_code, expected_exit)
                if sys.platform != "win32":
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
        self._assert_usage_exits_before_overlay(
            ["--not-an-option-" + attack], 2)
        with tempfile.TemporaryDirectory() as cwd:
            exit_code, output = self._run_cli(
                cwd, "--not-an-option-" + attack)

        self.assertEqual(exit_code, 2)
        self.assertNotIn(attack.encode(), output)
        if sys.platform != "win32":
            self.assertNotIn(b"\x1b", output)
        self.assertIn(b"\\x1b", output)


def _pty_child(argv):
    """Component-test child on the host's real PTY or ConPTY.

    ``mode-rollback`` and ``resize``/``resize-cancel`` verify native resource
    restoration and resize delivery. ``isig`` checks interrupt processing;
    ``cancel-flag``/``cancel-event`` report the effect of one input key;
    ``control-settings`` checks native settings through real input and actions.
    """
    import asyncio

    sys.path.insert(0, str(ROOT))
    from loki_agent import terminals as _terminals

    mode = argv[0]

    def terminal_state():
        if os.name == 'posix':
            return _terminals.termios.tcgetattr(0)
        native = _terminals.host_terminal_windows
        return (native._console_mode(native._handle(0)), native._GetConsoleCP(),
                native._console_mode(native._GetStdHandle(native.STD_OUTPUT_HANDLE)))

    if mode == 'capture':
        async def capture():
            with _terminals.TerminalMode(0, enabled=True):
                async with _terminals.AsyncKeyReader(0) as reader:
                    print('CAPTURE_READY', flush=True)
                    event = await reader.read_key()
                    assert event.kind == 'TEXT' and event.text == 'x', event
            # Exceed one capture read so draining must retain trailing bytes,
            # including the reset and end marker, before reporting exit status.
            sys.stdout.buffer.write(b'z' * 131072 + b'\x1b[0mCAPTURE_END\n')
            sys.stdout.buffer.flush()
        asyncio.run(capture())
        return 2

    if mode == 'control-settings':
        async def read_controls():
            before = terminal_state()
            try:
                if os.name == 'posix':
                    native = _terminals.termios
                    configured = native.tcgetattr(0)
                    for index, value in ((native.VERASE, b'\x15'), (native.VWERASE, b'\x16'),
                                         (native.VINTR, b'\x18')):
                        configured[6][index] = value
                    native.tcsetattr(0, native.TCSANOW, configured)
                # Read the actual host settings before raw mode, without
                # replacing either acquisition or the reader constructor.
                reader = _terminals.AsyncKeyReader(0)
                with _terminals.TerminalMode(0, enabled=True):
                    async with reader:
                        print('CONTROL_READY', flush=True)
                        events = [await reader.read_key() for _ in range(3)]
                        assert [event.kind for event in events] == ['BACKSPACE', 'BACKSPACE_WORD', 'CTRL_C'], events
                        assert reader.cancel_requested and reader.cancel_event.is_set()
            finally:
                if os.name == 'posix':
                    native.tcsetattr(0, native.TCSANOW, before)
            assert terminal_state() == before, 'control test changed terminal settings'
            print('CONTROL_ACTIONS_OK', flush=True)

        asyncio.run(read_controls())
        return 0

    if mode == 'mode-rollback':
        before = terminal_state()
        native = _terminals.termios if os.name == 'posix' else _terminals.host_terminal_windows
        name = 'tcsetattr' if os.name == 'posix' else '_SetConsoleMode'
        setter = getattr(native, name)
        first = True

        def fail_after_change(*args):
            nonlocal first
            result = setter(*args)
            if first:
                first = False
                assert terminal_state() != before, 'native setter did not change terminal state'
                raise OSError('injected failure after native mode change')
            return result

        with mock.patch.object(native, name, new=fail_after_change):
            try:
                _terminals.TerminalMode(0, enabled=True).__enter__()
            except OSError:
                pass
            else:
                raise AssertionError('mode setup did not fail')
        assert terminal_state() == before, 'failed entry changed terminal state'
        print('ROLLBACK_RESTORED', flush=True)
        return 0

    if mode in ['resize', 'resize-cancel']:
        async def resized():
            before = terminal_state()
            reader = _terminals.AsyncKeyReader(0, watch_resize=True, output_fd=1)
            stable = asyncio.Event()

            async def watch():
                with _terminals.TerminalMode(0, enabled=True):
                    async with reader:
                        print('RESIZE_READY', flush=True)
                        event = await reader.read_key()
                        assert event.kind == 'RESIZE', event
                        size = os.get_terminal_size(1)
                        assert size == (100, 35), size
                        print('RESIZE_RECEIVED', flush=True)
                        stable.set()
                        if mode == 'resize-cancel':
                            await asyncio.Future()

            task = asyncio.create_task(watch())
            if mode == 'resize-cancel':
                await stable.wait()
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                await task
            assert terminal_state() == before, 'exit changed terminal state'
            print('RESIZE_STOPPED', flush=True)
            # Parent resizes again, then acknowledges with Enter. A fresh
            # reader without resize watching lets the loop run without
            # reviving the closed reader, and decodes native newline forms.
            acknowledgment = _terminals.AsyncKeyReader(0)
            with _terminals.TerminalMode(0, enabled=True):
                async with acknowledgment:
                    event = await acknowledgment.read_key()
                    assert event.kind == 'ENTER', event
            assert terminal_state() == before, 'acknowledgment changed terminal settings'
            assert not reader.pending, reader.pending
            assert reader.byte_reader.queue.empty()
            print('RESIZE_QUIET', flush=True)

        asyncio.run(resized())
        return 0

    if mode == "isig":
        with _terminals.TerminalMode(0, enabled=True):
            sys.stdout.buffer.write(
                b"ISIG_SET\n"
                if _terminals.interrupt_processing_enabled(0)
                else b"ISIG_CLEAR\n")
            sys.stdout.buffer.flush()
        sys.stdout.buffer.write(
            b"RESTORED_ISIG_SET\n"
            if _terminals.interrupt_processing_enabled(0)
            else b"RESTORED_ISIG_CLEAR\n")
        sys.stdout.buffer.flush()
        return 0

    async def read_cancel(clear_event):
        reader = _terminals.AsyncKeyReader(0)
        with _terminals.TerminalMode(0, enabled=True):
            async with reader:
                if clear_event:
                    reader.cancel_event.clear()
                print('READER_READY', flush=True)
                key = await reader.read_key()
        return reader, key.kind

    if mode == "cancel-flag":
        async def main():
            reader, key_kind = await read_cancel(False)
            if reader.cancel_requested and key_kind == "CTRL_C":
                sys.stdout.buffer.write(b"CANCEL_SET\n")
            else:
                sys.stdout.buffer.write(
                    f"CANCEL_NOT_SET key={key_kind}\n".encode())
            sys.stdout.buffer.flush()
        asyncio.run(main())
        return 0

    if mode == "cancel-event":
        async def main():
            reader, key_kind = await read_cancel(True)
            if key_kind == "CTRL_C" and reader.cancel_event.is_set():
                sys.stdout.buffer.write(b"EVENT_SET\n")
            else:
                sys.stdout.buffer.write(b"EVENT_NOT_SET\n")
            sys.stdout.buffer.flush()
        asyncio.run(main())
        return 0

    raise SystemExit(f"unknown pty-child mode: {mode!r}")


class PtyResourceTests(unittest.TestCase):
    def read_marker(self, handle, output, marker):
        captured = bytearray(output)
        _read_until(handle, captured, marker.decode())
        return bytes(captured)

    def test_native_control_settings_drive_reader_actions(self):
        handle = pty_backend.spawn_pty([
            sys.executable, str(pathlib.Path(__file__).resolve()), '--pty-child', 'control-settings'])
        try:
            output = self.read_marker(handle, b'', b'CONTROL_READY')
            # POSIX has configurable control characters; the Windows console
            # uses its actual defaults. Both must drive the same reader actions.
            handle.write(b'\x15\x16\x18' if os.name == 'posix' else b'\x08\x17\x03')
            output = self.read_marker(handle, output, b'CONTROL_ACTIONS_OK')
            exit_code, output = _read_to_exit(handle, bytearray(output))
            self.assertEqual(exit_code, 0, output)
        finally:
            try:
                handle.terminate()
            finally:
                handle.close()

    def test_native_resize_and_shutdown_without_keyboard_input(self):
        for mode in ['resize', 'resize-cancel']:
            with self.subTest(mode=mode):
                handle = pty_backend.spawn_pty([
                    sys.executable, str(pathlib.Path(__file__).resolve()), '--pty-child', mode])
                try:
                    output = self.read_marker(handle, b'', b'RESIZE_READY')
                    handle.set_size(100, 35)
                    output = self.read_marker(handle, output, b'RESIZE_STOPPED')
                    self.assertIn(b'RESIZE_RECEIVED', output)
                    handle.set_size(110, 40)
                    handle.write(b'\r')
                    output = self.read_marker(handle, output, b'RESIZE_QUIET')
                    exit_code, output = _read_to_exit(handle, bytearray(output))
                    self.assertEqual(exit_code, 0, output)
                    self.assertNotIn(b'Traceback', output)
                finally:
                    try:
                        handle.terminate()
                    finally:
                        handle.close()

    def test_failed_mode_entry_restores_real_terminal_settings(self):
        handle = pty_backend.spawn_pty([
            sys.executable, str(pathlib.Path(__file__).resolve()), '--pty-child', 'mode-rollback'])
        try:
            output = self.read_marker(handle, b'', b'ROLLBACK_RESTORED')
            exit_code, output = _read_to_exit(handle, bytearray(output))
            self.assertEqual(exit_code, 0, output)
        finally:
            try:
                handle.terminate()
            finally:
                handle.close()


class PtyCtrlCTests(unittest.TestCase):
    """Ctrl+C must reach the reader as byte 0x03, not become SIGINT.

    With ISIG left set on the tty, the driver eats Ctrl+C and delivers
    SIGINT to the foreground process group before the byte reaches
    AsyncKeyReader, so cancel_requested never becomes True and a running
    turn cannot be cancelled between tool calls.  These tests pin the
    ISIG-clearing behavior of TerminalMode end to end.
    """

    def _spawn_child(self, mode):
        return pty_backend.spawn_pty(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--pty-child", mode])

    def test_terminal_mode_clears_isig(self):
        # Direct: enter TerminalMode on a fresh pty and inspect the flag.
        handle = self._spawn_child("isig")
        try:
            exit_code, out = _read_to_exit(handle)
            self.assertEqual(exit_code, 0, out)
        finally:
            try:
                handle.terminate()
            finally:
                handle.close()
        self.assertIn(b"ISIG_CLEAR", out)
        self.assertNotIn(b"ISIG_SET\n", out.replace(b"ISIG_CLEAR", b""))
        self.assertIn(b"RESTORED_ISIG_SET", out)

    def test_ctrl_c_sets_cancel_flag_in_reader(self):
        # End to end: with ISIG clear, a 0x03 byte written to the pty is
        # seen by AsyncKeyReader as CTRL_C and sets cancel_requested.
        handle = self._spawn_child("cancel-flag")
        try:
            output = bytearray()
            _read_until(handle, output, 'READER_READY')
            handle.write(b"\x03")
            exit_code, out = _read_to_exit(handle, output)
            self.assertEqual(exit_code, 0, out)
        finally:
            try:
                handle.terminate()
            finally:
                handle.close()
        self.assertIn(b"CANCEL_SET", out)
        self.assertNotIn(b"CANCEL_NOT_SET", out)


class PtyTurnCancelTests(unittest.TestCase):
    """A real turn: Ctrl+C must reach the reader's cancel event.

    The reader's cancel_event is what run_foreground races against to
    interrupt a foreground job; this pins the wiring from tty byte to
    event with the real InputSession machinery (per-turn clear included).
    """

    def test_ctrl_c_sets_reader_event_after_read_key(self):
        handle = pty_backend.spawn_pty(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--pty-child", "cancel-event"])
        try:
            output = bytearray()
            _read_until(handle, output, 'READER_READY')
            handle.write(b"\x03")
            exit_code, out = _read_to_exit(handle, output)
            self.assertEqual(exit_code, 0, out)
        finally:
            try:
                handle.terminate()
            finally:
                handle.close()
        self.assertIn(b"EVENT_SET", out)
        self.assertNotIn(b"EVENT_NOT_SET", out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--pty-child"]:
        raise SystemExit(_pty_child(sys.argv[2:]))
    unittest.main()
