import asyncio
import codecs
import collections
import fcntl
import os
import re
import signal
import sys
import termios
from dataclasses import dataclass


STATUS_COLOR = 4
INPUT_COLOR = 7
BOLD = '\033[1m'
CYAN = '\033[36m'
RESET = '\033[0m'


if not os.isatty(sys.stdout.fileno()):
  # Noninteractive tests and headless runs still import terminal helpers. The
  # no-op terminal keeps those paths from emitting escape sequences or touching
  # terminal state when stdout is not a TTY.
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
        print('\033[s', end='')

    def restore_cursor_position(self):
        print('\033[u', end='')

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

    def reset_colors_and_flags(self):
        print('\033[m', end='')

    def enable_bracketed_paste_mode(self): # \e[200~ ... \e[201~
        print('\033[?2004h', end='')

    def disable_bracketed_paste_mode(self):
        print('\033[?2004l', end='')

    def markdown_to_ansi(self, text: str) -> str:
        # We split the text by inline code blocks.
        # Using a capture group `(`.*?`)` ensures the code segments are kept in the resulting list.
        parts = re.split(r'(`.*?`)', text)
        for i, part in enumerate(parts):
            # Check if the current part is a code block
            if part.startswith('`') and part.endswith('`') and len(part) >= 2:
                inner_text = part[1:-1] # Strip the backticks
                parts[i] = f"{CYAN}{inner_text}{RESET}"
            else: # normal text
                 # .*? is non-greedy so it correctly matches isolated **bold** pairs
                part = re.sub(r'\*\*(.*?)\*\*', f'{BOLD}\\1{RESET}', part) # bold; TODO: also __foo__ also bold!
                part = re.sub(r'(?<!\*)\*(.*?)\*(?!\*)', f'\033[3m\\1{RESET}', part) # italics
                # TODO: maybe underline.
                # TODO: maybe strikethrough text: ~
                # TODO: # headline, ## headline, ### headline; also extra line ===== or ----
                # TODO: handle >blockquote blocks
                # TODO: ![Tux, the Linux mascot](/assets/images/tux.png)
                # TODO: \* Without the backslash, this would be a bullet in an unordered list.
                parts[i] = part

        return "".join(parts)


terminal = Terminal()

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
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.old_flags | os.O_NONBLOCK)
        self.loop.add_reader(self.fd, self._on_readable)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.loop is not None:
            self.loop.remove_reader(self.fd)
        if self.old_flags is not None:
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.old_flags)

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
            new_attrs = termios.tcgetattr(self.fd)
            new_attrs[3] &= ~(termios.ICANON | termios.ECHO)
            new_attrs[6][termios.VMIN] = 1
            new_attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSADRAIN, new_attrs)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)


KEY_SEQUENCES = {
    b'\x1b[A': "CURSOR_UP",
    b'\x1b[B': "CURSOR_DOWN",
    b'\x1b[C': "CURSOR_RIGHT",
    b'\x1b[D': "CURSOR_LEFT",
    b'\x1b[H': "HOME",
    b'\x1b[F': "END",
    b'\x1b[3~': "DELETE",
    b'\x1b[5~': "PAGE_UP",
    b'\x1b[6~': "PAGE_DOWN",
    b'\x1b[200~': "PASTE_START",
    b'\x1b[201~': "PASTE_END",
}
CSI_FINAL_BYTES = set(range(0x40, 0x7f))


class AsyncKeyReader:
    def __init__(self, fd: int, watch_resize: bool = False):
        self.fd = fd
        self.watch_resize = watch_resize
        self.byte_reader = AsyncByteReader(fd)
        self.pending = collections.deque()
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.escape = bytearray()
        self.paste_mode = False
        self.loop = None

    def _on_resize(self):
        self.pending.append(KeyEvent("RESIZE"))
        self.byte_reader.queue.put_nowait(b'')

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        await self.byte_reader.__aenter__()
        if self.watch_resize:
            try:
                self.loop.add_signal_handler(signal.SIGWINCH, self._on_resize)
            except (NotImplementedError, RuntimeError):
                # Some event loops/platforms do not expose signal handlers; the
                # prompt remains usable without live resize events.
                pass
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.watch_resize and self.loop is not None:
            try:
                self.loop.remove_signal_handler(signal.SIGWINCH)
            except (NotImplementedError, RuntimeError):
                # Match the add path: absence of signal-handler support is not
                # a terminal-state cleanup failure.
                pass
        await self.byte_reader.__aexit__(exc_type, exc, tb)

    def _emit_text_byte(self, byte: int):
        text = self.decoder.decode(bytes([byte]), final=False)
        if text:
            self.pending.append(KeyEvent("TEXT", text))

    def _feed_byte(self, byte: int):
        if self.escape:
            self.escape.append(byte)
            if self.escape == b'\x1b[':
                return
            if self.escape.startswith(b'\x1b[') and byte in CSI_FINAL_BYTES:
                sequence = bytes(self.escape)
                self.escape.clear()

                if byte == 0x52: # 0x52 is 'R'
                    # Only consume the exact two-parameter CPR reply that Loki
                    # asked for with DSR 6. Other CSI ... R sequences are not
                    # treated as cursor position guesses.
                    if re.match(br'^\x1b\[\d+;\d+R$', sequence):
                        self.pending.append(KeyEvent("CPR", sequence.decode('ascii')))
                        return

                kind = KEY_SEQUENCES.get(sequence)
                if kind == "PASTE_START":
                    self.paste_mode = True
                    self.pending.append(KeyEvent(kind))
                elif kind == "PASTE_END":
                    self.paste_mode = False
                    self.pending.append(KeyEvent(kind))
                elif kind:
                    self.pending.append(KeyEvent(kind))
                return
            if len(self.escape) == 2 and not self.escape.startswith(b'\x1b['):
                self.escape.clear()
                return
            if len(self.escape) > 32:
                # Unsupported escape sequences should not leave the input
                # parser stuck forever waiting for a final byte.
                self.escape.clear()
            return

        if byte == 0x1b:
            self.escape.append(byte)
        elif byte == 0x03:
            self.pending.append(KeyEvent("CTRL_C"))
        elif byte == 0x04:
            self.pending.append(KeyEvent("CTRL_D"))
        elif byte in [0x0a, 0x0d]:
            if self.paste_mode:
                self.pending.append(KeyEvent("TEXT", "\n"))
            else:
                self.pending.append(KeyEvent("ENTER"))
        elif byte in [0x7f, 0x08]:
            self.pending.append(KeyEvent("BACKSPACE"))
        else:
            self._emit_text_byte(byte)

    async def read_key(self) -> KeyEvent:
        while True:
            if self.pending:
                return self.pending.popleft()
            chunk = await self.byte_reader.read()
            if chunk == b'':
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
        for ch in text:
            self.chars.insert(self.cursor, ch)
            self.cursor += 1

    def backspace(self):
        if self.cursor > 0:
            del self.chars[self.cursor - 1]
            self.cursor -= 1

    def delete(self):
        if self.cursor < len(self.chars):
            del self.chars[self.cursor]

    def left(self):
        self.cursor = max(0, self.cursor - 1)

    def right(self):
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
        refresh_terminal_layout()
        self.terminal.set_clipping_region(*input_area)
        self.terminal.goto_position(1, 1)
        self.terminal.set_background_color(INPUT_COLOR)
        # ESC[J clears below the cursor, including the status area in this
        # layout. Redraw status immediately, then re-enter the input area.
        self.terminal.clear_to_end_of_screen()
        update_status_bar()
        self.terminal.set_clipping_region(*input_area)
        self.terminal.goto_position(1, 1)
        self.terminal.set_background_color(INPUT_COLOR)
        print(self.prompt + buffer.before_cursor(), end='')
        self.terminal.save_cursor_position()
        print(buffer.after_cursor(), end='')
        self.terminal.restore_cursor_position()
        self.terminal.flush()


class PromptController:
    def __init__(self, terminal, prompt: str = 'User: ', history=None):
        self.terminal = terminal
        self.prompt = prompt
        self.history = list(history or [])

    async def read_text(self) -> str:
        fd = sys.stdin.fileno()
        interactive = os.isatty(fd) and os.isatty(sys.stdout.fileno())
        buffer = InputBuffer()
        renderer = PromptRenderer(self.terminal, self.prompt)

        output_row, output_col = 1, 1
        history_index = len(self.history)
        saved_input = ""

        try:
            with TerminalMode(fd, interactive):
                async with AsyncKeyReader(fd, watch_resize=interactive) as reader:
                    if interactive:
                        sys.stdout.write('\033[6n')
                        sys.stdout.flush()

                        queued_events = []
                        while True:
                            event = await reader.read_key()
                            if event.kind == "CPR":
                                m = re.match(r'^\033\[(\d+);(\d+)R$', event.text)
                                if m:
                                    output_row = int(m.group(1))
                                    output_col = int(m.group(2))
                                break
                            elif event.kind != "EOF":
                                queued_events.append(event)

                        # Bytes typed before the terminal answers the CPR query
                        # still belong to the prompt, so replay them after the
                        # initial cursor-position handshake.
                        reader.pending.extendleft(reversed(queued_events))
                        renderer.render(buffer)

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
                                print()
                                self.terminal.flush()
                            return buffer.text()
                        if event.kind == "TEXT":
                            buffer.insert(event.text)
                        elif event.kind == "BACKSPACE":
                            buffer.backspace()
                        elif event.kind == "DELETE":
                            buffer.delete()
                        elif event.kind == "CURSOR_LEFT":
                            buffer.left()
                        elif event.kind == "CURSOR_RIGHT":
                            buffer.right()
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
                        elif event.kind in ["PASTE_START", "PASTE_END", "RESIZE"]:
                            # Paste markers only affect AsyncKeyReader state;
                            # resize is handled by the next render pass.
                            pass
                        if interactive:
                            renderer.render(buffer)
        finally:
            if interactive:
                # The prompt renderer temporarily owns the input/status regions;
                # put subsequent output back where the original prompt started.
                self.terminal.set_clipping_region(*output_area)
                self.terminal.goto_position(output_row, output_col)
                self.terminal.reset_colors_and_flags()
                self.terminal.flush()


async def get_input_async(prompt=None, history=None):
    return await PromptController(terminal, prompt or 'User: ', history=history).read_text()


def get_input(prompt=None, history=None):
    return asyncio.run(get_input_async(prompt, history=history))


async def run_menu_async(items):
    # Menus can be large--so show it in the output area.
    terminal.save_cursor_position()
    while True:
        terminal.restore_cursor_position()
        for i, item in enumerate(items):
            print(f"{i + 1}. {item}")
        selection = await get_input_async('User choice: ')
        selection = selection.strip()
        if selection.endswith('.'):
            selection = selection.rstrip('.')

        if selection:
            terminal.save_cursor_position()
            try:
                return items[int(selection) - 1]
            except (ValueError, IndexError):
                return selection

    terminal.save_cursor_position()
    return 'None of the above'


def run_menu(items):
    return asyncio.run(run_menu_async(items))


def restore_output_area_after_input():
    terminal.reset_colors_and_flags()
    terminal.flush()


def run_input_loop():
    while True:
        input_text = get_input()
        yield input_text
