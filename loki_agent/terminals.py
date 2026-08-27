import asyncio
import codecs
import collections
import contextlib
import fcntl
import os
import signal
import sys
import termios
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag


STATUS_COLOR = 4
INPUT_COLOR = 7
BOLD = '\033[1m'
ITALIC = '\033[3m'
CODE_COLOR = 6
RESET = '\033[0m'
MARKDOWN_MAX_UNRESOLVED = 4096
# Markdown headlines carry inline markup (`## The **uv** route`), but ANSI has
# no style stack: a plain RESET closes everything, and restoring "the rest of
# the state" would require tracking it. SGR channels are independent, so the
# headline owns the background channel (42 = green bg) while inline spans
# inside use foreground/attribute channels, each closed by its
# parameter-specific cancel (22 bold-off, 23 italic-off, 39 default fg,
# 49 default bg). That gives one nesting level with zero state tracking.
# Attribute-inside-attribute (`**b *i* b**`) is intentionally out of scope:
# it needs both channels' bookkeeping, i.e. the stack we refused.
HEADLINE_BG = 42
HEADLINE_BG_OFF = 49
HEADLINE = f'\033[{HEADLINE_BG}m'
HEADLINE_OFF = f'\033[{HEADLINE_BG_OFF}m'
BOLD_OFF = '\033[22m'
ITALIC_OFF = '\033[23m'
FOREGROUND_OFF = '\033[39m'


def open_terminal_stdin() -> int:
    """Own the keyboard: replace fd 0 with a fresh /dev/tty open.

    stdin and stdout normally share one open file description (the pty
    slave), so AsyncByteReader setting O_NONBLOCK on stdin for async
    reads would also make stdout non-blocking -- and print() then raises
    BlockingIOError (EAGAIN) when the kernel write buffer fills.
    Reopening /dev/tty as fd 0 gives reads a fresh open file description
    whose status flags are independent of stdout's.

    The terminal front-end calls this when it starts reading keys.
    Importing this module never touches fd 0: headless and ACP processes
    read stdin themselves and must not have it swapped under them.
    Without a controlling tty this is a no-op returning fd 0.
    """
    global new_stdin
    if not os.isatty(0):
        new_stdin = 0
        return new_stdin
    sys.stdin.close()
    new_stdin = os.open('/dev/tty', os.O_RDONLY)
    return new_stdin


new_stdin = sys.stdin.fileno()

if not os.isatty(new_stdin):
  # Noninteractive tests and headless runs still import terminal helpers. The
  # no-op terminal keeps those paths from emitting escape sequences or touching
  # terminal state when stdin is not a TTY.
  class Terminal:
      def __getattr__(self, x):
          return lambda *args, **kwargs: None
else:
  class Terminal:
    def __init__(self):
        self.bracketed_paste = False

    def clear_screen(self):
        print('\033[2J', end='')

    def clear_to_end_of_screen(self): # always, not relative.
        print('\033[J', end='')

    def goto_position(self, row, column):
        print('\033[{};{}H'.format(row, column), end='')

    def set_clipping_region(self, first_row, last_row): # note: after that, cursor position is (1,1) ABSOLUTE OR RELATIVE DEPENDING ON origin_mode
        assert last_row - first_row >= 2 # otherwise not supported.
        print('\033[{};{}r'.format(first_row, last_row - 1), end='')
        #goto_position(1, 1)

    def disable_clipping_regions(self): # note: after that, cursor position is (1,1) either absolute or relative dependig on origin_mode.
        print('\033[r', end='')
        #goto_position(1, 1)

    def save_cursor_position(self):
        print('\0337', end='')

    def restore_cursor_position(self):
        print('\0338', end='')

    def flush(self):
        sys.stdout.flush()

    def enable_origin_mode(self): # relative coordinates
        print('\033[?6h', end='')

    def disable_origin_mode(self):
        print('\033[?6l', end='')

    def set_foreground_color(self, index):
        print('\033[{}m'.format(30 + index), end='')

    def set_background_color(self, index):
        print('\033[{}m'.format(40 + index), end='')

    def set_reverse_video(self, enabled: bool):
        # 7 = reverse video (SGR); 27 = reverse off. Don't use 0 (full reset)
        # because that would clobber the input-area background color mid-caret.
        print('\033[7m' if enabled else '\033[27m', end='')

    def reset_colors_and_flags(self):
        print('\033[m', end='')

    def hide_cursor(self): # DECTCEM: the input area draws its own reverse-video caret.
        print('\033[?25l', end='')

    def show_cursor(self):
        print('\033[?25h', end='')

    def enable_bracketed_paste_mode(self): # \e[200~ ... \e[201~
        print('\033[?2004h', end='')

    def disable_bracketed_paste_mode(self):
        print('\033[?2004l', end='')

    def markdown_to_ansi(self, text: str) -> str:
        return markdown_line_to_ansi(text)


terminal = Terminal()


class BoundedMarkdownAnsi:
    """Incrementally render Loki's small Markdown dialect.

    Design target: a bounded, deterministic scanner that immediately emits
    coalesced decided prefixes, retains only genuinely unresolved Markdown,
    applies identical overflow semantics in batch and streaming modes, never
    leaves ANSI state active across event boundaries, and is tested for
    visibility before completion.

    Supported inline constructs are single-backtick code, ``*emphasis*``, and
    ``**bold**``. They never cross a newline and are styled only after their
    closing delimiter arrives. Backslash escapes and nested inline constructs
    are intentionally not interpreted. Fenced regions use matching runs of at
    least three backticks or tildes, with zero to three leading spaces; their
    contents and marker lines pass through verbatim.

    Headlines are one to six ``#`` at line start (after up to three leading
    spaces) followed by a space; the whole line renders on the SGR background
    channel, and inline spans inside it render on the foreground/attribute
    channels with parameter-specific resets, so both compose without a style
    stack. Nesting an attribute inside another attribute (``**b *i* b**``)
    stays literal as everywhere else; seven or more ``#``, or any ``#`` run
    not followed by a space, renders literally.

    ``feed`` and ``finish`` eagerly return tuples of terminal-state-neutral
    fragments. Plain text is emitted immediately. Only a possible fence or
    headline marker line, or an unresolved inline span, is retained, and none
    may exceed ``max_unresolved`` characters. After overflow, the remainder of
    that line is literal so a later closing delimiter cannot be reinterpreted
    as a new opener. Batch and streaming rendering use this same fallback
    rule.
    """

    def __init__(self, *, style=True,
                 max_unresolved=MARKDOWN_MAX_UNRESOLVED, inner=False):
        if (not isinstance(max_unresolved, int)
                or isinstance(max_unresolved, bool)
                or max_unresolved < 1):
            raise ValueError("max_unresolved must be a positive integer")
        self.style = bool(style)
        self.max_unresolved = max_unresolved
        # inner=True is the headline-body pass: it must never detect another
        # headline (a "#" here is content), which is also what bounds the
        # recursion in _emit_completed_span to exactly one level.
        self.inner = bool(inner)
        self._closed = False
        self._at_line_start = True
        self._fence_char = None
        self._fence_length = 0
        self._bol_pending = []
        self._inline_mode = "plain"
        self._inline_pending = []
        self._bold_star = False

    @property
    def retained_characters(self):
        return len(self._bol_pending) + len(self._inline_pending)

    @staticmethod
    def _emit(output, text):
        if text:
            output.append(text)

    @staticmethod
    def _fragments(output):
        rendered = "".join(output)
        return (rendered,) if rendered else ()

    def _reset_inline(self):
        self._inline_mode = "plain"
        self._inline_pending.clear()
        self._bold_star = False

    def _emit_pending_literal(self, output):
        self._emit(output, "".join(self._inline_pending))
        self._reset_inline()

    def _emit_completed_span(self, output):
        raw = "".join(self._inline_pending)
        mode = self._inline_mode
        if not self.style:
            rendered = raw
        elif mode == "bold":
            # RESET would also kill an open headline background, so inner
            # spans close with parameter-specific cancels instead.
            rendered = BOLD + raw[2:-2] + (BOLD_OFF if self.inner else RESET)
        elif mode == "emphasis":
            rendered = (
                ITALIC + raw[1:-1]
                + (ITALIC_OFF if self.inner else RESET))
        elif mode == "code":
            rendered = (
                f"\033[3{CODE_COLOR}m" + raw[1:-1]
                + (FOREGROUND_OFF if self.inner else RESET))
        elif mode == "headline":
            # The whole line is one span whose body is re-scanned for inline
            # spans by a nested inner pass. That pass can never produce a
            # headline (see __init__), so this recurses exactly one level
            # and stays bounded by max_unresolved.
            nested = BoundedMarkdownAnsi(
                style=self.style, max_unresolved=self.max_unresolved,
                inner=True)
            body = "".join(nested.feed(raw) + nested.finish())
            rendered = f"{HEADLINE}{body}{HEADLINE_OFF}"
        else:
            raise AssertionError(f"cannot render inline mode {mode!r}")
        self._emit(output, rendered)
        self._reset_inline()

    def _check_inline_bound(self, output):
        if len(self._inline_pending) > self.max_unresolved:
            self._emit_pending_literal(output)
            self._inline_mode = "literal_line"

    def _consume_inline(self, character, output):
        mode = self._inline_mode
        if mode == "literal_line":
            self._emit(output, character)
            if character == "\n":
                self._inline_mode = "plain"
                self._at_line_start = True
            return
        if mode == "plain":
            if character == "\n":
                self._emit(output, character)
                self._at_line_start = True
            elif character == "*":
                self._inline_mode = "star"
                self._inline_pending.append(character)
            elif character == "`":
                self._inline_mode = "code"
                self._inline_pending.append(character)
            elif self._at_line_start and not self.inner and character == "#":
                # Line-start only: a "#" mid-line ("a #2 pencil") is text.
                # Not in the inner pass: there the line already IS a
                # headline body, so "#" is content, never another marker.
                self._inline_mode = "hashtag"
                self._inline_pending.append(character)
            else:
                self._emit(output, character)
            return

        if mode == "hashtag":
            # Up to six "#" (CommonMark h6); a seventh is ordinary text and
            # falls out through the pending-literal path below.
            if character == "#" and len(self._inline_pending) < 6:
                self._inline_pending.append(character)
                return
            if character == " " and len(self._inline_pending) <= 6:
                # The space is part of the span: the background covers the
                # marker run as well.
                self._inline_pending.append(character)
                self._inline_mode = "headline"
                return
            # Not a headline. The run renders literally and the character
            # resumes ordinary inline scanning (so "##**b**" still bolds).
            # Clearing _at_line_start caps this at one headline attempt per
            # line start: a faux "#######" must not restart detection at
            # its seventh "#".
            self._at_line_start = False
            self._emit_pending_literal(output)
            self._consume_inline(character, output)
            return

        if mode == "star":
            if character == "*":
                self._inline_mode = "bold"
                self._inline_pending.append(character)
            elif character == "\n":
                self._emit_pending_literal(output)
                self._emit(output, character)
                self._at_line_start = True
            else:
                self._inline_mode = "emphasis"
                self._inline_pending.append(character)
                self._check_inline_bound(output)
            return

        if mode == "headline":
            if character == "\n":
                self._emit_completed_span(output)
                # Fallthrough
            else:
                # The retained headline span obeys the same bound as every
                # other span: past max_unresolved it flushes literally and
                # the line degrades to text.
                self._inline_pending.append(character)
                self._check_inline_bound(output)
                return

        if character == "\n":
            self._emit_pending_literal(output)
            self._emit(output, character)
            self._at_line_start = True
            return

        self._inline_pending.append(character)
        if mode == "emphasis":
            if character == "*":
                self._emit_completed_span(output)
            else:
                self._check_inline_bound(output)
            return
        if mode == "code":
            if character == "`":
                self._emit_completed_span(output)
            else:
                self._check_inline_bound(output)
            return
        if mode == "bold":
            if character == "*":
                if self._bold_star:
                    self._emit_completed_span(output)
                else:
                    self._bold_star = True
            else:
                self._bold_star = False
            if self._inline_mode == "bold":
                self._check_inline_bound(output)
            return

        raise AssertionError(f"unknown inline mode {mode!r}")

    @staticmethod
    def _leading_spaces(text):
        count = 0
        while count < len(text) and text[count] == " ":
            count += 1
        return count

    @classmethod
    def _opening_candidate(cls, text):
        spaces = cls._leading_spaces(text)
        if spaces > 3:
            return None
        rest = text[spaces:]
        if not rest:
            return ("possible", None, 0)
        marker = rest[0]
        if marker == "#":
            # Headlines share the BOL buffer with fence candidates: a bare
            # "#" run stays "possible" (one char of lookahead decides), and
            # any non-"#" after it settles the question, so the whole line
            # hands over to the inline scanner, whose hashtag mode owns the
            # marker-or-literal decision.
            run = 0
            while run < len(rest) and rest[run] == "#":
                run += 1
            if run == len(rest):
                return ("possible", None, 0)
            return None
        if marker not in ["`", "~"]:
            return None
        run = 0
        while run < len(rest) and rest[run] == marker:
            run += 1
        suffix = rest[run:]
        if not suffix:
            return ("possible", marker, run)
        if run >= 3:
            return ("open", marker, run)
        return None

    @classmethod
    def _closing_candidate(cls, text, marker, minimum):
        spaces = cls._leading_spaces(text)
        if spaces > 3:
            return (False, False)
        rest = text[spaces:]
        if not rest:
            return (True, False)
        run = 0
        while run < len(rest) and rest[run] == marker:
            run += 1
        if run == 0:
            return (False, False)
        suffix = rest[run:]
        if not suffix:
            return (True, run >= minimum)
        if run >= minimum and all(ch in " \t" for ch in suffix):
            return (True, True)
        return (False, False)

    def _consume_opening_line_start(self, character, output):
        if character == "\n":
            pending = "".join(self._bol_pending)
            state = self._opening_candidate(pending)
            if state is not None and state[1] is not None and state[2] >= 3:
                self._fence_char = state[1]
                self._fence_length = state[2]
                self._emit(output, pending + character)
            else:
                self._bol_pending.clear()
                # Feed before clearing _at_line_start: a flushed leading "#"
                # must still be recognized as a possible headline marker.
                for pending_character in pending:
                    self._consume_inline(pending_character, output)
                self._at_line_start = False
                self._consume_inline(character, output)
                return
            self._bol_pending.clear()
            self._at_line_start = True
            return

        self._bol_pending.append(character)
        pending = "".join(self._bol_pending)
        state = self._opening_candidate(pending)
        if state is not None and state[0] == "open":
            self._fence_char = state[1]
            self._fence_length = state[2]
            self._emit(output, pending)
            self._bol_pending.clear()
            self._at_line_start = False
            return
        if state is None:
            self._bol_pending.clear()
            # Feed before clearing _at_line_start: a flushed leading "#"
            # must still be recognized as a possible headline marker.
            for pending_character in pending:
                self._consume_inline(pending_character, output)
            self._at_line_start = False
            return
        if len(self._bol_pending) > self.max_unresolved:
            self._emit(output, pending)
            self._bol_pending.clear()
            self._at_line_start = False
            self._inline_mode = "literal_line"

    def _consume_fenced_line_start(self, character, output):
        if character == "\n":
            pending = "".join(self._bol_pending)
            _possible, closes = self._closing_candidate(
                pending, self._fence_char, self._fence_length)
            self._emit(output, pending + character)
            self._bol_pending.clear()
            if closes:
                self._fence_char = None
                self._fence_length = 0
            self._at_line_start = True
            return

        self._bol_pending.append(character)
        pending = "".join(self._bol_pending)
        possible, _closes = self._closing_candidate(
            pending, self._fence_char, self._fence_length)
        if not possible or len(self._bol_pending) > self.max_unresolved:
            self._emit(output, pending)
            self._bol_pending.clear()
            self._at_line_start = False

    def _consume(self, character, output):
        if self._at_line_start:
            if self._fence_char is None:
                self._consume_opening_line_start(character, output)
            else:
                self._consume_fenced_line_start(character, output)
            return
        if self._fence_char is not None:
            self._emit(output, character)
            if character == "\n":
                self._at_line_start = True
            return
        self._consume_inline(character, output)

    def feed(self, text):
        if self._closed:
            raise RuntimeError("cannot feed a finished Markdown renderer")
        if not isinstance(text, str):
            raise TypeError("Markdown chunks must be strings")
        if not self.style:
            return (text,) if text else ()
        output = []
        for character in text:
            self._consume(character, output)
        return self._fragments(output)

    def finish(self):
        if self._closed:
            return ()
        output = []
        if self._bol_pending:
            self._emit(output, "".join(self._bol_pending))
            self._bol_pending.clear()
        if self._inline_pending:
            self._emit_pending_literal(output)
        self._closed = True
        return self._fragments(output)


def _terminal_supports_markdown_ansi():
    return terminal.markdown_to_ansi("") is not None


def render_markdown(text, *, style=None,
                    max_unresolved=MARKDOWN_MAX_UNRESOLVED):
    """Render complete text through the same bounded scanner used live."""
    if style is None:
        style = _terminal_supports_markdown_ansi()
    renderer = BoundedMarkdownAnsi(
        style=style, max_unresolved=max_unresolved)
    return "".join(renderer.feed(text) + renderer.finish())


def markdown_line_to_ansi(text: str) -> str:
    """Compatibility entry point for explicitly ANSI-styled text."""
    return render_markdown(text, style=True)


class AssistantMarkdownPresentation:
    """Own one terminal assistant turn's Markdown-renderer lifecycle."""

    def __init__(self):
        self._renderer = None

    @property
    def active(self):
        return self._renderer is not None

    def start(self):
        stale = self.finish()
        self._renderer = BoundedMarkdownAnsi(
            style=_terminal_supports_markdown_ansi())
        return stale

    def feed(self, text):
        if self._renderer is None:
            self.start()
        return self._renderer.feed(text)

    def finish(self):
        if self._renderer is None:
            return ()
        renderer = self._renderer
        self._renderer = None
        return renderer.finish()

    def reset(self):
        self._renderer = None


terminal.assistant_markdown = AssistantMarkdownPresentation()


try:
    terminal_size = os.get_terminal_size()
    terminal_lines = terminal_size.lines
except OSError:
    terminal_lines = 25

terminal_lines = terminal_lines + 1 # last line in 1-based indices is missing otherwise.
output_area = 1, terminal_lines - 4
input_area = terminal_lines - 4, terminal_lines - 2
status_area = terminal_lines - 2, terminal_lines # too big, but that's the minimum supported height of set_clipping_region

'''
    output area
    question area
        regular question
        diff viewer (maybe huge with actual scrolling need!)
    input area
    tab to switch to next area
    status line
'''

_status_text_provider = lambda: ""


def set_status_text_provider(provider):
    global _status_text_provider
    _status_text_provider = provider


def update_status_bar():
    terminal.set_clipping_region(*status_area)
    terminal.goto_position(1, 1)
    terminal.set_background_color(STATUS_COLOR)
    terminal.clear_to_end_of_screen()
    print(_status_text_provider(), end='')


def redraw_status_bar():
    """Refresh only the status area while preserving the output cursor."""
    try:
        interactive = (
            os.isatty(new_stdin)
            and os.isatty(sys.stdout.fileno()))
    except (AttributeError, OSError, ValueError):
        interactive = False
    if not interactive:
        return

    refresh_terminal_layout()
    terminal.save_cursor_position()
    try:
        update_status_bar()
    finally:
        terminal.set_clipping_region(*output_area)
        terminal.restore_cursor_position()
        terminal.reset_colors_and_flags()
        terminal.flush()


def refresh_terminal_layout():
    global terminal_lines
    global output_area
    global input_area
    global status_area
    try:
        terminal_size = os.get_terminal_size()
        terminal_lines = terminal_size.lines + 1
    except OSError:
        terminal_lines = 25
    output_area = 1, terminal_lines - 4
    input_area = terminal_lines - 4, terminal_lines - 2
    status_area = terminal_lines - 2, terminal_lines


@dataclass
class KeyEvent:
    kind: str
    text: str = ""


class AsyncByteReader:
    def __init__(self, fd: int):
        self.fd = fd
        self.loop = None
        self.queue = asyncio.Queue()
        self.old_flags = None
        self._reader_registered = False

    def _on_readable(self):
        try:
            data = os.read(self.fd, 4096)
        except BlockingIOError:
            return
        except OSError as e:
            self.queue.put_nowait(e)
            return
        self.queue.put_nowait(data)

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        self.old_flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        try:
            fcntl.fcntl(
                self.fd, fcntl.F_SETFL, self.old_flags | os.O_NONBLOCK)
            self.loop.add_reader(self.fd, self._on_readable)
            self._reader_registered = True
        except BaseException:
            try:
                self._restore_flags()
            finally:
                self.loop = None
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._reader_registered and self.loop is not None:
                self.loop.remove_reader(self.fd)
                self._reader_registered = False
        finally:
            try:
                self._restore_flags()
            finally:
                self.loop = None

    def _restore_flags(self):
        if self.old_flags is None:
            return
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.old_flags)
        self.old_flags = None

    async def read(self):
        item = await self.queue.get()
        if isinstance(item, OSError):
            raise item
        return item


class TerminalMode:
    def __init__(self, fd: int, enabled: bool):
        self.fd = fd
        self.enabled = enabled
        self.old_attrs = None

    def __enter__(self):
        if self.enabled:
            self.old_attrs = termios.tcgetattr(self.fd)
            new_attrs = self.old_attrs.copy()
            new_attrs[6] = self.old_attrs[6].copy()
            # ISIG too: with it set, the tty driver turns Ctrl+C into SIGINT
            # for the foreground process group before the byte reaches the
            # reader, so AsyncKeyReader.cancel_requested can never fire and
            # between-tool-call cancellation never triggers.  Clearing ISIG
            # makes Ctrl+C the byte (0x03) the reader already handles.
            # Tool subprocesses are unaffected either way: they run in their
            # own session (start_new_session) and receive no tty signals.
            new_attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
            new_attrs[6][termios.VMIN] = 1
            new_attrs[6][termios.VTIME] = 0
            try:
                termios.tcsetattr(
                    self.fd, termios.TCSADRAIN, new_attrs)
            except BaseException:
                self.restore()
                raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()

    def restore(self):
        if self.old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
            self.old_attrs = None


PASTE_START_SEQUENCE = b'\x1b[200~'
PASTE_END_SEQUENCE = b'\x1b[201~'
FALLBACK_BACKSPACE_BYTES = frozenset((0x08, 0x7f))
FALLBACK_BACKSPACE_WORD_BYTES = frozenset((0x17,))
FALLBACK_INTERRUPT_BYTES = frozenset((0x03,))


def _control_character_byte(value):
    if isinstance(value, int):
        byte = value
    elif isinstance(value, (bytes, bytearray)) and len(value) == 1:
        byte = value[0]
    elif isinstance(value, str) and len(value) == 1:
        byte = ord(value)
    else:
        return None
    if not 0 <= byte <= 0xff:
        return None
    return byte


def terminal_control_bytes(fd: int):
    """Return configured character-erase, word-erase, and interrupt bytes."""
    try:
        [
            iflag,
            oflag,
            cflag,
            lflag,
            ispeed,
            ospeed,
            cc,
        ] = termios.tcgetattr(fd)
    except (termios.error, OSError, TypeError, ValueError):
        return (
            FALLBACK_BACKSPACE_BYTES,
            FALLBACK_BACKSPACE_WORD_BYTES,
            FALLBACK_INTERRUPT_BYTES,
        )

    # POSIX lets each tty choose the byte that disables a control character.
    try:
        disabled_byte = os.fpathconf(fd, 'PC_VDISABLE')
    except (OSError, TypeError, ValueError):
        disabled = {0x00, 0xff}
    else:
        if isinstance(disabled_byte, int) and 0 <= disabled_byte <= 0xff:
            disabled = {disabled_byte}
        else:
            disabled = {0x00, 0xff}

    def configured(index, fallback):
        try:
            byte = _control_character_byte(cc[index])
        except (IndexError, TypeError):
            byte = None
        if byte is None or byte in disabled:
            return fallback
        return frozenset((byte,))

    try:
        backspace_bytes = configured(
            termios.VERASE, FALLBACK_BACKSPACE_BYTES)
    except AttributeError:
        backspace_bytes = FALLBACK_BACKSPACE_BYTES
    try:
        backspace_word_bytes = configured(
            termios.VWERASE, FALLBACK_BACKSPACE_WORD_BYTES)
    except AttributeError:
        backspace_word_bytes = FALLBACK_BACKSPACE_WORD_BYTES
    try:
        interrupt_bytes = configured(
            termios.VINTR, FALLBACK_INTERRUPT_BYTES)
    except AttributeError:
        interrupt_bytes = FALLBACK_INTERRUPT_BYTES
    return backspace_bytes, backspace_word_bytes, interrupt_bytes


TERMINFO_KEY_CAPABILITIES = (
    ("kcuu1", "CURSOR_UP"),
    ("kcud1", "CURSOR_DOWN"),
    ("kcuf1", "CURSOR_RIGHT"),
    ("kcub1", "CURSOR_LEFT"),
    ("khome", "HOME"),
    ("kend", "END"),
    ("kdch1", "DELETE"),
    ("kpp", "PAGE_UP"),
    ("knp", "PAGE_DOWN"),
    ("kLFT5", "CURSOR_WORD_LEFT"),
    ("kRIT5", "CURSOR_WORD_RIGHT"),
    ("kcbt", "MODE_CYCLE"),
)
MAX_KEY_SEQUENCE_BYTES = 32
CSI_PARAMETER_MIN = 0x30
CSI_PARAMETER_MAX = 0x3f
CSI_INTERMEDIATE_MIN = 0x20
CSI_INTERMEDIATE_MAX = 0x2f
CSI_FINAL_MIN = 0x40
CSI_FINAL_MAX = 0x7e
KEY_READER_WAKE = object()


class KittyModifier(IntFlag):
    SHIFT = 1
    ALT = 2
    CTRL = 4
    SUPER = 8
    HYPER = 16
    META = 32
    CAPS_LOCK = 64
    NUM_LOCK = 128


KITTY_LOCK_MODIFIERS = (
    KittyModifier.CAPS_LOCK | KittyModifier.NUM_LOCK)
KITTY_ALL_MODIFIERS = (
    KittyModifier.SHIFT
    | KittyModifier.ALT
    | KittyModifier.CTRL
    | KittyModifier.SUPER
    | KittyModifier.HYPER
    | KittyModifier.META
    | KittyModifier.CAPS_LOCK
    | KittyModifier.NUM_LOCK
)


class KittyEventType(IntEnum):
    """Wire values from Kitty's modifiers:event-type field."""

    PRESS = 1
    REPEAT = 2
    RELEASE = 3


class KittyKeyCode(IntEnum):
    TAB = 9
    ENTER = 13
    ESCAPE = 27
    BACKSPACE = 127
    KP_ENTER = 57414
    KP_LEFT = 57417
    KP_RIGHT = 57418
    KP_UP = 57419
    KP_DOWN = 57420
    KP_PAGE_UP = 57421
    KP_PAGE_DOWN = 57422
    KP_HOME = 57423
    KP_END = 57424
    KP_DELETE = 57426


def terminfo_key_sequences(output_fd=None):
    """Return optional key encodings without making curses a dependency."""
    try:
        import curses

        if output_fd is None:
            output_fd = sys.stdout.fileno()
        curses.setupterm(fd=output_fd)
    except Exception:
        return {}

    sequences = {}
    for capability, kind in TERMINFO_KEY_CAPABILITIES:
        try:
            sequence = curses.tigetstr(capability)
        except Exception:
            continue
        # Input capabilities should be control sequences. Refuse values which
        # could capture ordinary text, consume unbounded parser state, or
        # cannot be decoded by Loki's ESC-prefixed input path.
        if (not isinstance(sequence, bytes)
                or not 2 <= len(sequence) <= MAX_KEY_SEQUENCE_BYTES
                or not sequence.startswith(b'\x1b')):
            continue
        sequences.setdefault(sequence, kind)
    return sequences


@dataclass(frozen=True)
class ControlSequence:
    introducer: str
    parameters: bytes
    intermediates: bytes
    final: int
    raw: bytes


@dataclass(frozen=True)
class EscapeParseResult:
    sequence: ControlSequence | None = None
    replay: bytes = b""


@dataclass
class EscapeSequenceParser:
    """Frame bounded ESC-prefixed terminal input without timing guesses."""

    exact_sequences: frozenset[bytes] = frozenset()
    max_bytes: int = MAX_KEY_SEQUENCE_BYTES
    state: str = field(default="ground", init=False)
    raw: bytearray = field(default_factory=bytearray, init=False)
    parameters: bytearray = field(default_factory=bytearray, init=False)
    intermediates: bytearray = field(default_factory=bytearray, init=False)
    exact_prefixes: frozenset[bytes] = field(init=False)

    def __post_init__(self):
        self.exact_sequences = frozenset(self.exact_sequences)
        self.exact_prefixes = frozenset(
            sequence[:length]
            for sequence in self.exact_sequences
            for length in range(2, len(sequence))
        )

    @property
    def active(self):
        return self.state != "ground"

    def begin(self):
        self._reset()
        self.state = "escape"
        self.raw.append(0x1b)

    def _reset(self):
        self.state = "ground"
        self.raw.clear()
        self.parameters.clear()
        self.intermediates.clear()

    @staticmethod
    def _is_final(byte):
        return CSI_FINAL_MIN <= byte <= CSI_FINAL_MAX

    def _discard(self, byte):
        # Once framing is invalid, retain no attacker-controlled payload.
        # A final byte still terminates the malformed sequence atomically.
        self.raw.clear()
        self.parameters.clear()
        self.intermediates.clear()
        if self._is_final(byte):
            self.state = "ground"
        else:
            self.state = "discard"

    def _complete(self, introducer):
        sequence = ControlSequence(
            introducer=introducer,
            parameters=bytes(self.parameters),
            intermediates=bytes(self.intermediates),
            final=self.raw[-1],
            raw=bytes(self.raw),
        )
        self._reset()
        return EscapeParseResult(sequence=sequence)

    def feed(self, byte):
        if not self.active:
            raise RuntimeError("escape parser is not active")

        if self.state == "discard":
            if self._is_final(byte):
                self._reset()
            return None

        if len(self.raw) >= self.max_bytes:
            self._discard(byte)
            return None
        self.raw.append(byte)
        raw = bytes(self.raw)

        if self.state == "escape":
            if byte == 0x1b:
                # With no timeout, another ESC is the only unambiguous way to
                # restart an otherwise standalone Escape prefix.
                self.raw[:] = b"\x1b"
                return None
            if byte == ord("["):
                self.state = "csi_parameters"
                return None
            if byte == ord("O"):
                self.state = "ss3"
                return None
            if raw in self.exact_sequences:
                return self._complete("TERMINFO")
            if raw in self.exact_prefixes:
                self.state = "terminfo"
                return None

            replay = bytes((byte,))
            self._reset()
            return EscapeParseResult(replay=replay)

        if self.state == "terminfo":
            if raw in self.exact_sequences:
                return self._complete("TERMINFO")
            if raw in self.exact_prefixes:
                return None

            replay = raw[1:]
            self._reset()
            return EscapeParseResult(replay=replay)

        if self.state == "ss3":
            if self._is_final(byte):
                return self._complete("SS3")
            self._discard(byte)
            return None

        if CSI_PARAMETER_MIN <= byte <= CSI_PARAMETER_MAX:
            if self.state != "csi_parameters":
                self._discard(byte)
                return None
            self.parameters.append(byte)
            return None
        if CSI_INTERMEDIATE_MIN <= byte <= CSI_INTERMEDIATE_MAX:
            self.state = "csi_intermediates"
            self.intermediates.append(byte)
            return None
        if self._is_final(byte):
            return self._complete("CSI")

        self._discard(byte)
        return None


@dataclass(frozen=True)
class KittyKey:
    code: int
    shifted_code: int | None
    base_layout_code: int | None
    modifiers: KittyModifier
    event_type: KittyEventType
    text: tuple[int, ...] | None


def _numeric_parameter_fields(parameters):
    if not parameters:
        return ()
    if any(
            not (
                ord("0") <= byte <= ord("9")
                or byte in (ord(";"), ord(":"))
            )
            for byte in parameters):
        return None

    fields = []
    for field_bytes in parameters.split(b";"):
        subfields = []
        for value in field_bytes.split(b":"):
            subfields.append(None if not value else int(value))
        fields.append(tuple(subfields))
    return tuple(fields)


def _modifier_field(field):
    if not 1 <= len(field) <= 2:
        return None
    # Kitty adds one to the modifier bit field, so wire value 1 means that
    # no modifier is active. The optional colon subfield is the event type.
    encoded_modifiers = 1 if field[0] is None else field[0]
    event_value = (
        KittyEventType.PRESS
        if len(field) == 1 or field[1] is None
        else field[1]
    )
    if not 1 <= encoded_modifiers <= int(KITTY_ALL_MODIFIERS) + 1:
        return None
    try:
        event_type = KittyEventType(event_value)
    except ValueError:
        return None
    return KittyModifier(encoded_modifiers - 1), event_type


def _functional_parameters(parameters):
    """Decode Kitty's CSI 1;modifier:event functional-key prefix."""
    fields = _numeric_parameter_fields(parameters)
    if fields is None:
        return None
    if not fields:
        return KittyModifier(0), KittyEventType.PRESS
    if fields == ((1,),):
        return KittyModifier(0), KittyEventType.PRESS
    if len(fields) != 2 or fields[0] != (1,):
        return None
    return _modifier_field(fields[1])


def _tilde_parameters(parameters):
    fields = _numeric_parameter_fields(parameters)
    if fields is None or not 1 <= len(fields) <= 2:
        return None
    if len(fields[0]) != 1 or fields[0][0] is None:
        return None
    if len(fields) == 1:
        modifiers = KittyModifier(0)
        event_type = KittyEventType.PRESS
    else:
        decoded = _modifier_field(fields[1])
        if decoded is None:
            return None
        modifiers, event_type = decoded
    return fields[0][0], modifiers, event_type


def _unicode_scalar(code):
    return (
        isinstance(code, int)
        and 0 <= code <= 0x10ffff
        and not 0xd800 <= code <= 0xdfff
    )


def _kitty_text_scalar(code):
    return (
        _unicode_scalar(code)
        and code >= 0x20
        and not 0x7f <= code <= 0x9f
    )


def parse_kitty_key(sequence):
    if (
            sequence.introducer != "CSI"
            or sequence.final != ord("u")
            or sequence.intermediates):
        return None

    fields = _numeric_parameter_fields(sequence.parameters)
    if fields is None or not 1 <= len(fields) <= 3:
        return None

    key_field = fields[0]
    if (
            not 1 <= len(key_field) <= 3
            or key_field[0] is None
            or not _unicode_scalar(key_field[0])):
        return None
    alternate_codes = list(key_field[1:]) + [None, None]
    shifted_code, base_layout_code = alternate_codes[:2]
    if (
            shifted_code is not None
            and not _unicode_scalar(shifted_code)):
        return None
    if (
            base_layout_code is not None
            and not _unicode_scalar(base_layout_code)):
        return None

    modifiers = KittyModifier(0)
    event_type = KittyEventType.PRESS
    if len(fields) >= 2:
        decoded = _modifier_field(fields[1])
        if decoded is None:
            return None
        modifiers, event_type = decoded

    text = None
    if len(fields) == 3:
        if not fields[2] or any(
                code is None or not _kitty_text_scalar(code)
                for code in fields[2]):
            return None
        text = tuple(fields[2])

    return KittyKey(
        code=key_field[0],
        shifted_code=shifted_code,
        base_layout_code=base_layout_code,
        modifiers=modifiers,
        event_type=event_type,
        text=text,
    )


CSI_FINAL_KEY_KINDS = {
    ord("A"): "CURSOR_UP",
    ord("B"): "CURSOR_DOWN",
    ord("C"): "CURSOR_RIGHT",
    ord("D"): "CURSOR_LEFT",
    ord("H"): "HOME",
    ord("F"): "END",
}
CSI_TILDE_KEY_KINDS = {
    3: "DELETE",
    5: "PAGE_UP",
    6: "PAGE_DOWN",
    7: "HOME",
    8: "END",
}
KITTY_KEYPAD_KEY_KINDS = {
    KittyKeyCode.KP_ENTER: "ENTER",
    KittyKeyCode.KP_LEFT: "CURSOR_LEFT",
    KittyKeyCode.KP_RIGHT: "CURSOR_RIGHT",
    KittyKeyCode.KP_UP: "CURSOR_UP",
    KittyKeyCode.KP_DOWN: "CURSOR_DOWN",
    KittyKeyCode.KP_PAGE_UP: "PAGE_UP",
    KittyKeyCode.KP_PAGE_DOWN: "PAGE_DOWN",
    KittyKeyCode.KP_HOME: "HOME",
    KittyKeyCode.KP_END: "END",
    KittyKeyCode.KP_DELETE: "DELETE",
}


def decode_legacy_control_sequence(sequence):
    """Return (recognized, event) for Loki's supported CSI/SS3 keys."""
    if sequence.introducer == "SS3":
        kind = CSI_FINAL_KEY_KINDS.get(sequence.final)
        return (kind is not None, KeyEvent(kind) if kind else None)
    if sequence.introducer != "CSI" or sequence.intermediates:
        return False, None

    if sequence.final in CSI_FINAL_KEY_KINDS:
        decoded = _functional_parameters(sequence.parameters)
        if decoded is None:
            return False, None
        modifiers, event_type = decoded
        modifiers &= ~KITTY_LOCK_MODIFIERS
        if event_type == KittyEventType.RELEASE:
            return True, None

        kind = CSI_FINAL_KEY_KINDS[sequence.final]
        if not modifiers:
            return True, KeyEvent(kind)
        if (
                modifiers == KittyModifier.CTRL
                and kind == "CURSOR_LEFT"):
            return True, KeyEvent("CURSOR_WORD_LEFT")
        if (
                modifiers == KittyModifier.CTRL
                and kind == "CURSOR_RIGHT"):
            return True, KeyEvent("CURSOR_WORD_RIGHT")
        return True, None

    if sequence.final == ord("~"):
        decoded = _tilde_parameters(sequence.parameters)
        if decoded is None:
            return False, None
        number, modifiers, event_type = decoded
        if (
                number in (200, 201)
                and not modifiers
                and event_type == KittyEventType.PRESS):
            kind = "PASTE_START" if number == 200 else "PASTE_END"
            return True, KeyEvent(kind)

        kind = CSI_TILDE_KEY_KINDS.get(number)
        if kind is None:
            return False, None
        modifiers &= ~KITTY_LOCK_MODIFIERS
        if event_type == KittyEventType.RELEASE or modifiers:
            return True, None
        return True, KeyEvent(kind)

    if sequence.final == ord("Z"):
        decoded = _functional_parameters(sequence.parameters)
        if decoded is None:
            return False, None
        modifiers, event_type = decoded
        modifiers &= ~KITTY_LOCK_MODIFIERS
        # CSI Z is the historical no-parameter form. Kitty's structured form
        # spells the same Shift-Tab key as CSI 1;2 Z.
        if not sequence.parameters:
            modifiers = KittyModifier.SHIFT
        if (
                event_type == KittyEventType.RELEASE
                or modifiers != KittyModifier.SHIFT):
            return True, None
        return True, KeyEvent("MODE_CYCLE")

    if sequence.final == ord("R"):
        fields = _numeric_parameter_fields(sequence.parameters)
        if (
                fields is not None
                and len(fields) == 2
                and all(
                    len(field) == 1 and field[0] is not None
                    for field in fields
                )):
            return True, KeyEvent(
                "CPR", sequence.raw.decode("ascii"))
        # Never reinterpret an ambiguous CSI R as a function key.
        return True, None

    return False, None


def _kitty_control_byte(code):
    if ord("a") <= code <= ord("z"):
        return code - ord("a") + 1
    if ord("A") <= code <= ord("Z"):
        return code - ord("A") + 1
    return {
        ord(" "): 0,
        ord("2"): 0,
        ord("@"): 0,
        ord("3"): 27,
        ord("["): 27,
        ord("4"): 28,
        ord("\\"): 28,
        ord("5"): 29,
        ord("]"): 29,
        ord("6"): 30,
        ord("^"): 30,
        ord("~"): 30,
        ord("7"): 31,
        ord("/"): 31,
        ord("_"): 31,
        ord("8"): 127,
        ord("?"): 127,
    }.get(code)


class AsyncKeyReader:
    def __init__(self, fd: int, watch_resize: bool = False,
                 use_terminfo: bool = False, output_fd=None):
        self.fd = fd
        self.watch_resize = watch_resize
        (
            self.backspace_bytes,
            self.backspace_word_bytes,
            self.interrupt_bytes,
        ) = terminal_control_bytes(fd)
        self.key_sequences = {}
        if use_terminfo:
            self.key_sequences.update(terminfo_key_sequences(output_fd))
        self.escape_parser = EscapeSequenceParser(
            frozenset(self.key_sequences))
        self.byte_reader = AsyncByteReader(fd)
        self.pending = collections.deque()
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.paste_mode = False
        self.paste_bytes = bytearray()
        self.eof = False
        self.loop = None
        self._resources = None
        self.cancel_requested = False
        # Event-driven counterpart of cancel_requested.  Set in _feed_byte,
        # which runs inside read_key(), which runs as a task on the same
        # asyncio loop as everything that waits on this event -- so a plain
        # .set() is sufficient (no call_soon_threadsafe needed: the setter
        # is loop context, not a foreign thread).
        self.cancel_event = asyncio.Event()
        # Shift-Tab: requests cycling the agent mode (explore/plan/edit) for the
        # next turn. Same mechanism as cancel_requested (a flag the main loop
        # reads), but it must NOT cancel -- the current turn keeps running.
        self.mode_cycle_requested = False

    def _on_resize(self):
        self.pending.append(KeyEvent("RESIZE"))
        self.byte_reader.queue.put_nowait(KEY_READER_WAKE)

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        resources = contextlib.AsyncExitStack()
        try:
            await resources.enter_async_context(self.byte_reader)
            if self.watch_resize:
                try:
                    self.loop.add_signal_handler(
                        signal.SIGWINCH, self._on_resize)
                except (NotImplementedError, RuntimeError):
                    # Some event loops/platforms do not expose signal
                    # handlers; the prompt remains usable without live resize
                    # events.
                    pass
                else:
                    resources.callback(self._remove_resize_handler)
        except BaseException:
            try:
                await resources.__aexit__(*sys.exc_info())
            finally:
                self.loop = None
            raise
        self._resources = resources
        return self

    async def __aexit__(self, exc_type, exc, tb):
        resources = self._resources
        self._resources = None
        try:
            if resources is not None:
                await resources.__aexit__(exc_type, exc, tb)
        finally:
            self.loop = None

    def _remove_resize_handler(self):
        if self.loop is None:
            return
        try:
            self.loop.remove_signal_handler(signal.SIGWINCH)
        except (NotImplementedError, RuntimeError):
            # Match the add path: absence of signal-handler support is not a
            # terminal-state cleanup failure.
            pass

    def _emit_text_byte(self, byte: int):
        text = self.decoder.decode(bytes([byte]), final=False)
        if text:
            self.pending.append(KeyEvent("TEXT", text))

    def _finish_paste(self):
        raw = bytes(self.paste_bytes)
        self.paste_bytes.clear()
        self.paste_mode = False
        if raw:
            # Match the existing input convention: carriage returns pasted
            # from a CR-oriented clipboard are input newlines.
            text = raw.decode("utf-8", errors="replace").replace("\r", "\n")
            self.pending.append(KeyEvent("TEXT", text))
        self.pending.append(KeyEvent("PASTE_END"))

    def _queue_event(self, event):
        if event.kind == "PASTE_START":
            self.paste_mode = True
            self.paste_bytes.clear()
        elif event.kind == "MODE_CYCLE":
            self.mode_cycle_requested = True
        elif event.kind == "CTRL_C":
            self.cancel_requested = True
            self.cancel_event.set()
        self.pending.append(event)

    def _feed_plain_byte(self, byte):
        if byte in self.interrupt_bytes:
            self._queue_event(KeyEvent("CTRL_C"))
        elif byte == 0x04:
            self._queue_event(KeyEvent("CTRL_D"))
        elif byte in (0x0a, 0x0d):
            self._queue_event(KeyEvent("ENTER"))
        elif byte in self.backspace_bytes:
            self._queue_event(KeyEvent("BACKSPACE"))
        elif byte in self.backspace_word_bytes:
            self._queue_event(KeyEvent("BACKSPACE_WORD"))
        else:
            self._emit_text_byte(byte)

    def _kitty_control_is_supported(self, byte):
        return (
            byte in self.interrupt_bytes
            or byte == 0x04
            or byte in (0x09, 0x0a, 0x0d)
            or byte in self.backspace_bytes
            or byte in self.backspace_word_bytes
        )

    def _handle_kitty_key(self, key):
        if key.event_type == KittyEventType.RELEASE:
            return
        if key.text is not None:
            self._queue_event(KeyEvent(
                "TEXT", "".join(chr(code) for code in key.text)))
            return

        modifiers = key.modifiers & ~KITTY_LOCK_MODIFIERS
        if (
                modifiers & KittyModifier.CTRL
                and not modifiers & ~(
                    KittyModifier.CTRL | KittyModifier.SHIFT)):
            for code in (key.code, key.base_layout_code):
                if code is None:
                    continue
                byte = _kitty_control_byte(code)
                if (
                        byte is not None
                        and self._kitty_control_is_supported(byte)):
                    self._feed_plain_byte(byte)
                    return

        kind = KITTY_KEYPAD_KEY_KINDS.get(key.code)
        if kind is not None:
            if not modifiers:
                self._queue_event(KeyEvent(kind))
            elif (
                    modifiers == KittyModifier.CTRL
                    and kind == "CURSOR_LEFT"):
                self._queue_event(KeyEvent("CURSOR_WORD_LEFT"))
            elif (
                    modifiers == KittyModifier.CTRL
                    and kind == "CURSOR_RIGHT"):
                self._queue_event(KeyEvent("CURSOR_WORD_RIGHT"))
            return

        if key.code == KittyKeyCode.ENTER and not modifiers:
            self._queue_event(KeyEvent("ENTER"))
            return
        if key.code == KittyKeyCode.BACKSPACE and not modifiers:
            self._queue_event(KeyEvent("BACKSPACE"))
            return
        if key.code == KittyKeyCode.TAB:
            if not modifiers:
                self._queue_event(KeyEvent("TEXT", "\t"))
            elif modifiers == KittyModifier.SHIFT:
                self._queue_event(KeyEvent("MODE_CYCLE"))
            return

        text_code = None
        if modifiers in (KittyModifier(0), KittyModifier.ALT):
            text_code = key.code
        elif modifiers in (
                KittyModifier.SHIFT,
                KittyModifier.SHIFT | KittyModifier.ALT):
            text_code = key.shifted_code
            if (
                    text_code is None
                    and ord("a") <= key.code <= ord("z")):
                text_code = ord(chr(key.code).upper())
        if (
                text_code is not None
                and not 57344 <= text_code <= 63743
                and _kitty_text_scalar(text_code)):
            self._queue_event(KeyEvent("TEXT", chr(text_code)))

    def _handle_control_sequence(self, sequence):
        if sequence.introducer != "TERMINFO":
            recognized, event = decode_legacy_control_sequence(sequence)
            if recognized:
                if event is not None:
                    self._queue_event(event)
                return
            if (
                    sequence.introducer == "CSI"
                    and sequence.final == ord("u")):
                key = parse_kitty_key(sequence)
                if key is not None:
                    self._handle_kitty_key(key)
                return

        kind = self.key_sequences.get(sequence.raw)
        if kind is not None:
            self._queue_event(KeyEvent(kind))

    def _feed_byte(self, byte: int):
        if self.paste_mode:
            self.paste_bytes.append(byte)
            if self.paste_bytes.endswith(PASTE_END_SEQUENCE):
                del self.paste_bytes[-len(PASTE_END_SEQUENCE):]
                self._finish_paste()
            return

        if self.escape_parser.active:
            result = self.escape_parser.feed(byte)
            if result is None:
                return
            if result.sequence is not None:
                self._handle_control_sequence(result.sequence)
            else:
                for replay_byte in result.replay:
                    self._feed_byte(replay_byte)
            return

        if byte == 0x1b:
            self.escape_parser.begin()
        else:
            self._feed_plain_byte(byte)

    async def read_key(self) -> KeyEvent:
        while True:
            if self.pending:
                return self.pending.popleft()
            if self.eof:
                return KeyEvent("EOF")
            chunk = await self.byte_reader.read()
            if chunk is KEY_READER_WAKE:
                continue
            if chunk == b'':
                self.eof = True
                if self.paste_mode:
                    self._finish_paste()
                if self.pending:
                    return self.pending.popleft()
                return KeyEvent("EOF")
            for byte in chunk:
                self._feed_byte(byte)


class InputBuffer:
    def __init__(self):
        self.chars = []
        self.cursor = 0

    def text(self) -> str:
        return ''.join(self.chars)

    def before_cursor(self) -> str:
        return ''.join(self.chars[:self.cursor])

    def after_cursor(self) -> str:
        return ''.join(self.chars[self.cursor:])

    def insert(self, text: str):
        self.chars[self.cursor:self.cursor] = text
        self.cursor += len(text)

    def backspace(self):
        if self.cursor > 0:
            del self.chars[self.cursor - 1]
            self.cursor -= 1

    def backspace_word(self):
        end = self.cursor
        self.word_left()
        del self.chars[self.cursor:end]

    def delete(self):
        if self.cursor < len(self.chars):
            del self.chars[self.cursor]

    def left(self):
        self.cursor = max(0, self.cursor - 1)

    def right(self):
        self.cursor = min(len(self.chars), self.cursor + 1)

    def word_left(self):
        if not self.chars or self.cursor <= 0:
            self.cursor = 0
            return
        position = min(self.cursor, len(self.chars))
        while position > 0 and not self.chars[position - 1].isidentifier():
            position -= 1
        while position > 0 and self.chars[position - 1].isidentifier():
            position -= 1
        self.cursor = position

    def word_right(self):
        if self.cursor >= 0 and self.cursor < len(self.chars):
            if self.chars[self.cursor].isidentifier():
                while self.cursor < len(self.chars):
                    self.cursor += 1
                    if self.cursor < len(self.chars):
                        if not self.chars[self.cursor].isidentifier():
                             break

            while self.cursor < len(self.chars) and not self.chars[self.cursor].isidentifier():
                self.cursor += 1
        else:
            self.cursor = min(len(self.chars), self.cursor + 1)

    def home(self):
        self.cursor = 0

    def end(self):
        self.cursor = len(self.chars)


class PromptRenderer:
    def __init__(self, terminal, prompt: str):
        self.terminal = terminal
        self.prompt = prompt

    def render(self, buffer: InputBuffer):
        """Draw the input area and status area, atomically.

        DECSC at entry snapshots the output cursor (and SGR, origin mode,
        ...). The try/finally guarantees that even on a mid-render
        exception (e.g. set_clipping_region's assert on a too-small
        resize, or a write I/O error) the scroll region is restored to
        output_area and the cursor is DECRC'd back to its snapshot. No
        awaits, so the event loop's no-interleave property makes this
        atomic w.r.t. output writers on the same loop.
        """
        refresh_terminal_layout()
        self.terminal.save_cursor_position()
        try:
            # Draw the input area and status area.
            self.terminal.set_clipping_region(*input_area)
            self.terminal.goto_position(1, 1)
            self.terminal.set_background_color(INPUT_COLOR)
            self.terminal.clear_to_end_of_screen()
            update_status_bar()
            self.terminal.set_clipping_region(*input_area)
            self.terminal.goto_position(1, 1)
            self.terminal.set_background_color(INPUT_COLOR)
            print(self.prompt + buffer.before_cursor(), end='')

            # Draw the fake (reverse-video) caret at the insertion point,
            # then the text after it.
            after = buffer.after_cursor()
            self.terminal.set_reverse_video(True)
            if after:
                print(after[0], end='')
            else:
                print(' ', end='')
            self.terminal.set_reverse_video(False)
            if after:
                print(after[1:], end='')
        finally:
            # Restore the output scroll region and the cursor position.
            # The real cursor is back in the output area where output
            # writers expect it -- whether the try block succeeded or not.
            self.terminal.set_clipping_region(*output_area)
            self.terminal.restore_cursor_position()
            self.terminal.reset_colors_and_flags()
            self.terminal.flush()


class PromptController:
    def __init__(self, terminal, prompt: str = 'User: ', history=None, session=None,
                 on_mode_cycle=None):
        self.terminal = terminal
        self.prompt = prompt
        self.history = list(history or [])
        self.session = session  # AsyncKeyReader held for the whole session
        self.on_mode_cycle = on_mode_cycle or (lambda: None)

    async def read_text(self) -> str:
        fd = new_stdin
        interactive = os.isatty(fd) and os.isatty(sys.stdout.fileno())
        buffer = InputBuffer()
        renderer = PromptRenderer(self.terminal, self.prompt)

        output_row, output_col = 1, 1
        history_index = len(self.history)
        saved_input = ""

        if self.session is not None:
            # Session owns raw mode + the reader for the whole session; the
            # prompt just consumes keys from it.
            return await self._read_text_from_reader(self.session, interactive, buffer, renderer,
                                                     output_row, output_col, history_index, saved_input)
        # No session: manage our own raw mode + reader for this one call.
        reader = AsyncKeyReader(
            fd,
            watch_resize=interactive,
            use_terminfo=interactive,
            output_fd=sys.stdout.fileno() if interactive else None,
        )
        with TerminalMode(fd, interactive):
            async with reader:
                return await self._read_text_from_reader(reader, interactive, buffer, renderer,
                                                         output_row, output_col, history_index, saved_input)

    async def _read_text_from_reader(self, reader, interactive, buffer, renderer,
                                     output_row, output_col, history_index, saved_input):
        if interactive:
            renderer.render(buffer)

        paste_active = False
        while True:
            event = await reader.read_key()
            if event.kind == "EOF":
                if buffer.text():
                    return buffer.text()
                raise EOFError
            if event.kind == "CTRL_C":
                raise KeyboardInterrupt
            if event.kind == "CTRL_D":
                if buffer.text():
                    return buffer.text()
                raise EOFError
            if event.kind == "ENTER":
                if interactive:
                    #print() This would unnecessarily scroll
                    self.terminal.flush()
                return buffer.text()
            if event.kind == "TEXT":
                buffer.insert(event.text)
            elif event.kind == "BACKSPACE":
                buffer.backspace()
            elif event.kind == "BACKSPACE_WORD":
                buffer.backspace_word()
            elif event.kind == "DELETE":
                buffer.delete()
            elif event.kind == "CURSOR_LEFT":
                buffer.left()
            elif event.kind == "CURSOR_RIGHT":
                buffer.right()
            elif event.kind == "CURSOR_WORD_LEFT":
                buffer.word_left()
            elif event.kind == "CURSOR_WORD_RIGHT":
                buffer.word_right()
            elif event.kind == "HOME":
                buffer.home()
            elif event.kind == "END":
                buffer.end()
            elif event.kind in ["CURSOR_UP", "PAGE_UP"]:
                if self.history and history_index > 0:
                    if history_index == len(self.history):
                        saved_input = buffer.text()
                    history_index -= 1
                    buffer = InputBuffer()
                    buffer.insert(self.history[history_index])
            elif event.kind in ["CURSOR_DOWN", "PAGE_DOWN"]:
                if history_index < len(self.history):
                    history_index += 1
                    buffer = InputBuffer()
                    if history_index == len(self.history):
                        buffer.insert(saved_input)
                    else:
                        buffer.insert(self.history[history_index])
            elif event.kind == "PASTE_START":
                paste_active = True
            elif event.kind == "PASTE_END":
                paste_active = False
            elif event.kind == "RESIZE":
                # Resize is handled by the next render pass. If one arrives
                # during a paste, PASTE_END supplies that render.
                pass
            elif event.kind == "MODE_CYCLE":
                # Shift-Tab cycles the agent mode; does not cancel or alter
                # the buffer.
                self.on_mode_cycle()
            if interactive and not paste_active:
                renderer.render(buffer)


async def get_input_async(prompt=None, history=None, session=None, on_mode_cycle=None):
    return await PromptController(terminal, prompt or 'User: ', history=history, session=session,
                                  on_mode_cycle=on_mode_cycle).read_text()


class InputModal:
    """Exclusive modal-input path for one InputSession.

    Entering suspends the normal user-message producer. While active, prompt()
    is the only supported consumer of the session's reader. Exiting restores
    the normal producer.
    """

    def __init__(self, session):
        self.session = session
        self.active = False
        self.reading = False

    async def __aenter__(self):
        if self.session._modal is not None:
            raise RuntimeError("an input modal is already active")
        self.session._modal = self
        try:
            await self.session._pause()
        except BaseException:
            self.session._modal = None
            raise
        self.active = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.active = False
        if self.session._modal is self:
            self.session._modal = None
        await self.session._resume()

    async def prompt(self, prompt_text='User: ', history=None) -> str:
        if not self.active or self.session._modal is not self:
            raise RuntimeError("modal prompt used outside its active context")
        if self.reading:
            raise RuntimeError("modal already has an active prompt")
        self.reading = True
        try:
            return await get_input_async(
                prompt_text, history, session=self.session.reader)
        finally:
            self.reading = False


class UserMessageQueue(asyncio.Queue):
    """Queue that reports submitted, non-sentinel messages explicitly."""

    def __init__(self, on_size_change=None):
        super().__init__()
        self._message_count = 0
        self._on_size_change = on_size_change or (lambda count: None)

    @property
    def message_count(self) -> int:
        return self._message_count

    def _notify_size_change(self):
        try:
            self._on_size_change(self._message_count)
        except Exception as error:
            # A status-display callback must never lose an input that has
            # already entered or left the queue.
            try:
                asyncio.get_running_loop().call_exception_handler({
                    "message": "user-message queue status callback failed",
                    "exception": error,
                })
            except RuntimeError:
                pass

    def put_nowait(self, item):
        super().put_nowait(item)
        if item is not None:
            self._message_count += 1
            self._notify_size_change()

    def get_nowait(self):
        item = super().get_nowait()
        if item is not None:
            assert self._message_count > 0
            self._message_count -= 1
            self._notify_size_change()
        return item

    def discard_pending_messages(self):
        old_count = self._message_count
        while True:
            try:
                super().get_nowait()
            except asyncio.QueueEmpty:
                break
        self._message_count = 0
        if old_count:
            self._notify_size_change()


class InputSession:
    """Session-long input owner: raw mode, one stdin reader, producer, queue.

    Everything that touches the terminal input lives here, so the rest of the
    program never constructs a reader or flips raw mode itself.  Usage::

        async with input_session() as session:
            # session.user_messages: queue of submitted lines (the producer
            # enqueues; the turn loop drains).
            # async with session.modal() as modal: suspends that producer and
            # gives the modal exclusive access through modal.prompt(...).

    Raw mode and the reader are held from enter to exit, so keys typed while
    the agent works are drained into the queue instead of being lost to the
    kernel's line buffer.  Ctrl+C in raw mode is a CTRL_C key event; the
    reader sets ``cancel_requested`` and the turn loop checks it between
    awaits (the old SIGINT emergency stop no longer exists).
    """

    def __init__(self, fd=None, on_mode_cycle=None, history_provider=None,
                 on_queue_size_change=None):
        self.fd = fd if fd is not None else new_stdin
        self.interactive = os.isatty(self.fd) and os.isatty(sys.stdout.fileno())
        self.reader = AsyncKeyReader(
            self.fd,
            watch_resize=self.interactive,
            use_terminfo=self.interactive,
            output_fd=sys.stdout.fileno() if self.interactive else None,
        )
        self.user_messages = UserMessageQueue(on_queue_size_change)
        self._producer = None
        self._mode = None
        self._modal = None
        self._resources = None
        self.on_mode_cycle = on_mode_cycle or (lambda: None)
        self.history_provider = history_provider

    async def __aenter__(self):
        resources = contextlib.AsyncExitStack()
        try:
            self._mode = resources.enter_context(
                TerminalMode(self.fd, self.interactive))
            await resources.enter_async_context(self.reader)
            self._producer = asyncio.create_task(self._produce())
        except BaseException:
            try:
                await resources.__aexit__(*sys.exc_info())
            finally:
                self._mode = None
            raise
        self._resources = resources
        return self

    async def __aexit__(self, exc_type, exc, tb):
        resources = self._resources
        self._resources = None
        try:
            try:
                if self._producer is not None:
                    self._producer.cancel()
                    try:
                        await self._producer
                    except (asyncio.CancelledError, EOFError):
                        pass
                    finally:
                        self._producer = None
            finally:
                self.user_messages.discard_pending_messages()
        finally:
            try:
                if resources is not None:
                    await resources.__aexit__(*sys.exc_info())
            finally:
                self._mode = None

    async def _produce(self):
        while True:
            try:
                history = self.history_provider() if self.history_provider else None
                text = await get_input_async(session=self.reader, history=history,
                                             on_mode_cycle=self.on_mode_cycle)
            except EOFError:
                self.user_messages.put_nowait(None)  # sentinel: end of session
                return
            except KeyboardInterrupt:
                # Ctrl+C at the prompt cancels the current prompt; keep looping
                # so the user can keep typing. cancel_requested is set by the
                # reader.
                continue
            except Exception as e:
                # An unexpected exception (e.g. render failing on a tiny
                # resize, or a write I/O error) would otherwise kill the
                # producer task silently and leave the main loop blocked on
                # user_messages.get() forever. Log and treat it as session EOF
                # so the main loop breaks and main()'s clean_up runs.
                import traceback
                print(f"input session error: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                self.user_messages.put_nowait(None)
                return
            self.user_messages.put_nowait(text)

    def modal(self) -> InputModal:
        """Return the sole modal path for temporarily consuming terminal input."""
        return InputModal(self)

    async def _pause(self):
        if self._producer is not None:
            self._producer.cancel()
            try:
                await self._producer
            except (asyncio.CancelledError, EOFError):
                pass
            self._producer = None

    async def _resume(self):
        if self._producer is None:
            self._producer = asyncio.create_task(self._produce())


def input_session(fd=None, on_mode_cycle=None, history_provider=None,
                  on_queue_size_change=None) -> InputSession:
    return InputSession(
        fd,
        on_mode_cycle=on_mode_cycle,
        history_provider=history_provider,
        on_queue_size_change=on_queue_size_change,
    )


def restore_output_area_after_input():
    terminal.reset_colors_and_flags()
    terminal.flush()
