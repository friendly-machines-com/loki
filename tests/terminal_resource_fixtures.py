"""Host-native environments for the shared terminal ownership tests."""

from contextlib import ExitStack, contextmanager
import copy
import os
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
