"""Host-native environments for the shared terminal ownership tests."""

from contextlib import ExitStack, contextmanager
import copy
import os
import threading
from unittest import mock

from loki_agent import terminals


class ModeEnvironment:
    """Stateful native-call boundary; TerminalMode and its backend stay real."""

    def __init__(self):
        self.failures = set()
        self.writes = []

    def write(self, key, value):
        phase = 'restore' if value == self.initial[key] else 'setup'
        self.writes.append((key, phase))
        if (key, phase, 'before') in self.failures:
            raise OSError(f'injected {key} {phase}')
        self.state[key] = copy.deepcopy(value)
        if (key, phase, 'after') in self.failures:
            raise OSError(f'injected {key} {phase}')
        if (key, phase, 'reported') in self.failures:
            if os.name == 'nt':
                return 0  # native BOOL failure, not a Python exception
            raise OSError(f'injected {key} {phase}')
        return 1

    @contextmanager
    def installed(self):
        with ExitStack() as stack:
            if os.name == 'posix':
                termios = terminals.termios
                cc = [b'\0'] * (max(termios.VMIN, termios.VTIME) + 1)
                self.initial = {'input': [0, 0, 0, termios.ICANON | termios.ECHO | termios.ISIG, 0, 0, cc]}
                stack.enter_context(mock.patch.object(
                    termios, 'tcgetattr', side_effect=lambda fd: copy.deepcopy(self.state['input'])))
                stack.enter_context(mock.patch.object(
                    termios, 'tcsetattr', side_effect=lambda fd, when, value: self.write('input', value)))
            else:
                import ctypes
                from ctypes import wintypes
                native = terminals.host_terminal_windows
                self.initial = {'input': 7, 'codepage': 437, 'output': 0}

                def get_mode(handle, result):
                    value = getattr(handle, 'value', handle)
                    ctypes.cast(result, ctypes.POINTER(wintypes.DWORD))[0] = self.state[
                        'input' if value == 41 else 'output']
                    return 1

                def set_mode(handle, value):
                    handle = getattr(handle, 'value', handle)
                    return self.write('input' if handle == 41 else 'output', value)

                replacements = {
                    '_handle': lambda fd: 41,
                    '_GetStdHandle': lambda which: 42,
                    '_GetConsoleMode': get_mode,
                    '_SetConsoleMode': set_mode,
                    '_GetConsoleCP': lambda: self.state['codepage'],
                    '_SetConsoleCP': lambda value: self.write('codepage', value),
                }
                for name, replacement in replacements.items():
                    stack.enter_context(mock.patch.object(native, name, new=replacement))
            self.state = copy.deepcopy(self.initial)
            yield self


class ReaderEnvironment:
    """Real pipe/loop, with failures injected at native acquisition/release."""

    def __init__(self, loop):
        self.loop = loop
        self.failures = set()
        self.registered = False
        self.threads = []
        self.events = set()
        self.exit_gate = threading.Event()
        self.exit_gate.set()

    @contextmanager
    def installed(self):
        self.read_fd, self.write_fd = os.pipe()
        with ExitStack() as stack:
            if os.name == 'posix':
                native = terminals.fcntl
                fcntl = native.fcntl
                self.original_flags = fcntl(self.read_fd, native.F_GETFL)
                add = self.loop.add_reader
                remove = self.loop.remove_reader

                def change_flags(fd, command, *values):
                    if fd == self.read_fd and command == native.F_SETFL:
                        phase = 'restore' if values[0] == self.original_flags else 'flags'
                        if phase in self.failures:
                            raise OSError('injected ' + phase)
                    return fcntl(fd, command, *values)

                def register(fd, *args):
                    result = add(fd, *args)
                    if fd == self.read_fd:
                        self.registered = True
                        if 'register' in self.failures:
                            raise OSError('injected register')
                    return result

                def unregister(fd):
                    if fd == self.read_fd and 'stop' in self.failures:
                        raise OSError('injected stop')
                    result = remove(fd)
                    if fd == self.read_fd:
                        self.registered = False
                    return result

                stack.enter_context(mock.patch.object(native, 'fcntl', new=change_flags))
                stack.enter_context(mock.patch.object(self.loop, 'add_reader', new=register))
                stack.enter_context(mock.patch.object(self.loop, 'remove_reader', new=unregister))
                self.acquisitions = ('flags', 'register')
                self.releases = ('stop', 'restore')
            else:
                from loki_agent import handle_reader as native
                create, close, signal = native._CreateEventW, native._CloseHandle, native._SetEvent
                thread_class = threading.Thread

                def create_event(*args):
                    if 'event' in self.failures:
                        return 0
                    event = create(*args)
                    if event:
                        self.events.add(event)
                    return event

                def close_event(event):
                    if 'close' in self.failures:
                        return 0
                    result = close(event)
                    if result:
                        self.events.discard(event)
                    return result

                def signal_event(event):
                    if 'signal' in self.failures:
                        return 0
                    return signal(event)

                def make_thread(*args, **kwargs):
                    if 'thread' in self.failures:
                        raise RuntimeError('injected thread')
                    target = kwargs['target']

                    def run():
                        target()
                        self.exit_gate.wait()

                    kwargs['target'] = run
                    thread = thread_class(*args, **kwargs)
                    start, join = thread.start, thread.join

                    def start_thread():
                        if 'start' in self.failures:
                            raise RuntimeError('injected start')
                        start()
                        if 'started' in self.failures:
                            raise RuntimeError('injected started')

                    def join_thread(timeout=None):
                        if not self.failures.intersection(('stop', 'signal')):
                            join(timeout)

                    thread.start = start_thread
                    thread.join = join_thread
                    self.threads.append((thread, join))
                    return thread

                for name, replacement in (('_CreateEventW', create_event), ('_CloseHandle', close_event),
                                          ('_SetEvent', signal_event)):
                    stack.enter_context(mock.patch.object(native, name, new=replacement))
                stack.enter_context(mock.patch.object(native.threading, 'Thread', new=make_thread))
                self.acquisitions = ('event', 'thread', 'start', 'started')
                self.releases = ('stop', 'signal', 'close')
            try:
                yield self
            finally:
                # Backstop after assertions; never close a borrowed pipe while
                # a failing implementation has a thread still reading it.
                self.failures.clear()
                self.exit_gate.set()
                if os.name == 'posix':
                    remove(self.read_fd)
                    fcntl(self.read_fd, native.F_SETFL, self.original_flags)
                else:
                    for event in self.events:
                        signal(event)
                    for thread, join in self.threads:
                        if thread.ident is not None:
                            join(3)
                            if thread.is_alive():
                                raise RuntimeError('fixture reader did not stop')
                    for event in tuple(self.events):
                        close(event)
                os.close(self.write_fd)
                os.close(self.read_fd)

    def assert_released(self, case):
        os.fstat(self.read_fd)  # the reader borrows, rather than owns, this fd
        if os.name == 'posix':
            case.assertFalse(self.registered)
            case.assertEqual(terminals.fcntl.fcntl(self.read_fd, terminals.fcntl.F_GETFL),
                             self.original_flags)
        else:
            case.assertFalse(self.events)
            case.assertTrue(all(not thread.is_alive() for thread, _ in self.threads))
