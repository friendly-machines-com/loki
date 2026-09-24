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
import time
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
        if c in (0x5D, 0x50, 0x58, 0x5E, 0x5F):  # OSC/DCS/SOS/PM/APC
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


def _read_with_timeout(handle, total=4.0):
    """Drain the pty; reset the deadline on each arriving chunk."""
    buf = b""
    deadline = time.time() + total
    while time.time() < deadline:
        remaining = max(0.05, deadline - time.time())
        chunk = handle.read(65536, min(0.2, remaining))
        if not chunk:
            if buf:
                break  # quiesced
            continue
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
    collected = b""
    before_stream_release = b""
    try:
        configure_container(env, tmpdir)
        handle = pty_backend.spawn_pty(
            [entrypoint("loki")], env=env, cwd=tmpdir)
        collected += _read_with_timeout(handle, 6.0)  # startup banner

        handle.write(initial_input + b"\r")
        reply_output = _read_with_timeout(handle, 4.0)
        collected += reply_output
        if gate:
            before_stream_release = reply_output
            for queued_input in queued_inputs or []:
                handle.write(queued_input.encode() + b"\r")
                queued_output = _read_with_timeout(handle, 2.0)
                collected += queued_output
                before_stream_release += queued_output
            pathlib.Path(gate).touch()
            collected += _read_with_timeout(handle, 4.0)

        handle.write(b"/quit\r")
        collected += _read_with_timeout(handle, 2.0)
    finally:
        try:
            if handle is not None:
                handle.terminate()
                handle.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return collected, before_stream_release


class PtyFixtureTests(unittest.TestCase):
    def test_private_trees_are_outside_the_granted_workspace(self):
        module = sys.modules[__name__]
        for cli in (False, True):
            with self.subTest(cli=cli):
                configured = []

                def configure(env, workspace):
                    self.assertTrue(os.path.isdir(workspace))
                    for key in ('XDG_CONFIG_HOME', 'XDG_STATE_HOME'):
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
                    mock.patch.object(module, '_read_with_timeout', return_value=b''),
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
            stream=True,
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
            rb"Local: CWD: [^\r\n]*, turn: idle, queued messages: 0, "
            rb"queued images: 0, mode: normal;",
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

        # These assertions concern queue transitions, independently of styling.
        before_release = re.sub(rb"\x1b\[[0-9;]*m", b"", before_release)
        output = re.sub(rb"\x1b\[[0-9;]*m", b"", output)
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

        # These assertions concern queue transitions, independently of styling.
        before_release = re.sub(rb"\x1b\[[0-9;]*m", b"", before_release)
        output = re.sub(rb"\x1b\[[0-9;]*m", b"", output)
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
        output = b""
        exit_code = None
        try:
            output = _read_with_timeout(handle, 4.0)
            # Bounded wait: a usage-path regression that waits for input
            # instead of exiting must FAIL the test, not hang the suite.
            for _ in range(50):  # up to 5s
                exit_code = handle.poll()
                if exit_code is not None:
                    break
                time.sleep(0.1)
        finally:
            handle.terminate()
            handle.close()
        if exit_code is None:
            self.fail(
                f"loki{tuple(cli_args)!r} never exited on the usage path; "
                f"output captured: {output!r}")
        return exit_code, output

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
    """Entrypoint the Ctrl+C tests spawn on the pty.

    Modes: ``isig`` reports whether interrupt processing is enabled before and
    after ``TerminalMode``; ``cancel-flag`` / ``cancel-event`` read one key and
    report whether it set ``cancel_requested`` / ``cancel_event``.
    """
    import asyncio

    sys.path.insert(0, str(ROOT))
    from loki_agent import terminals as _terminals

    mode = argv[0]

    if mode == "isig":
        tmode = _terminals.TerminalMode(0, enabled=True)
        tmode.__enter__()
        sys.stdout.buffer.write(
            b"ISIG_SET\n"
            if _terminals.interrupt_processing_enabled(0)
            else b"ISIG_CLEAR\n")
        sys.stdout.buffer.flush()
        tmode.__exit__(None, None, None)
        sys.stdout.buffer.write(
            b"RESTORED_ISIG_SET\n"
            if _terminals.interrupt_processing_enabled(0)
            else b"RESTORED_ISIG_CLEAR\n")
        sys.stdout.buffer.flush()
        time.sleep(0.5)
        return 0

    async def read_cancel(clear_event):
        tmode = _terminals.TerminalMode(0, enabled=True)
        tmode.__enter__()
        reader = _terminals.AsyncKeyReader(0)
        key_kind = None
        async with reader:
            if clear_event:
                reader.cancel_event.clear()
            try:
                key = await asyncio.wait_for(reader.read_key(), timeout=3.0)
                key_kind = key.kind
            except asyncio.TimeoutError:
                pass
        tmode.__exit__(None, None, None)
        return reader, key_kind

    if mode == "cancel-flag":
        async def main():
            reader, key_kind = await read_cancel(False)
            if reader.cancel_requested and key_kind == "CTRL_C":
                sys.stdout.buffer.write(b"CANCEL_SET\n")
            else:
                sys.stdout.buffer.write(
                    f"CANCEL_NOT_SET key={key_kind}\n".encode())
            sys.stdout.buffer.write(
                b"CANCEL_SET\n" if reader.cancel_requested
                else b"CANCEL_NOT_SET\n")
            sys.stdout.buffer.flush()
            time.sleep(0.3)
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
            time.sleep(0.2)
        asyncio.run(main())
        return 0

    raise SystemExit(f"unknown pty-child mode: {mode!r}")


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
            out = _read_with_timeout(handle, 4.0)
        finally:
            handle.terminate()
            handle.close()
        self.assertIn(b"ISIG_CLEAR", out)
        self.assertNotIn(b"ISIG_SET\n", out.replace(b"ISIG_CLEAR", b""))
        self.assertIn(b"RESTORED_ISIG_SET", out)

    def test_ctrl_c_sets_cancel_flag_in_reader(self):
        # End to end: with ISIG clear, a 0x03 byte written to the pty is
        # seen by AsyncKeyReader as CTRL_C and sets cancel_requested.
        handle = self._spawn_child("cancel-flag")
        try:
            _read_with_timeout(handle, 1.0)  # let child reach its wait
            handle.write(b"\x03")
            out = _read_with_timeout(handle, 4.0)
        finally:
            handle.terminate()
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
            _read_with_timeout(handle, 1.0)
            handle.write(b"\x03")
            out = _read_with_timeout(handle, 4.0)
        finally:
            handle.terminate()
            handle.close()
        self.assertIn(b"EVENT_SET", out)
        self.assertNotIn(b"EVENT_NOT_SET", out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--pty-child"]:
        raise SystemExit(_pty_child(sys.argv[2:]))
    unittest.main()
