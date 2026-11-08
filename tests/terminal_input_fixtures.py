"""Keyboard-driven prompt tests with the real editor, producer and modal handoff.

Ask, history and provider-seed tests need to interrupt an unfinished edit at a
known key boundary. Native resource fixtures exercise acquisition failures,
not editable prompts; stubbing get_input_async would bypass the draft lifetime
under test. This fixture substitutes only terminal resources, rendering and
key delivery, acknowledging each read after the preceding key was processed.
"""

import asyncio
from contextlib import contextmanager, redirect_stdout
import io
from unittest import mock

from process_lifecycle_fixtures import observe_fixture
from loki_agent import terminals


class PromptKeyboard:
    def __init__(self):
        self.keys = asyncio.Queue()
        self.reads = asyncio.Queue()
        self.frames = []
        self.cancel_event = asyncio.Event()
        self.reading = False
        self.eof = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def read_key(self):
        if self.reading:
            raise AssertionError('two prompts are consuming keyboard input')
        self.reading = True
        try:
            self.reads.put_nowait(self.frames[-1] if self.frames else None)
            event = terminals.KeyEvent('EOF') if self.eof else await self.keys.get()
            if event.kind == 'CTRL_C':
                self.cancel_event.set()
            elif event.kind == 'EOF':
                self.eof = True
            return event
        finally:
            self.reading = False

    def feed(self, kind, text=''):
        self.keys.put_nowait(terminals.KeyEvent(kind, text))

    async def next_read(self, owner):
        return await observe_fixture(self.reads.get(), owner)

    async def press(self, owner, kind, text=''):
        self.feed(kind, text)
        return await self.next_read(owner)

    @contextmanager
    def installed(self, *, interactive=True):
        keyboard = self

        class Output(io.StringIO):
            def fileno(self):
                return 1

        class Renderer:
            def __init__(self, terminal, prompt):
                self.prompt = prompt

            def render(self, buffer):
                keyboard.frames.append([self.prompt, buffer.text(), buffer.cursor])

        with (
            mock.patch.object(terminals, 'AsyncKeyReader', return_value=self),
            mock.patch.object(terminals, 'TerminalMode'),
            mock.patch.object(terminals, 'open_terminal_stdin'),
            mock.patch.object(terminals.os, 'isatty', return_value=interactive),
            mock.patch.object(terminals, 'PromptRenderer', Renderer),
            redirect_stdout(Output()),
        ):
            yield self
