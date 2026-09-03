"""In-memory startup credentials with process-environment scrubbing."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterable, MutableMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .authentications import CredentialBroker, CredentialRef


CREDENTIAL_SUFFIXES = ("_KEY", "_TOKEN", "_PAT")


def is_credential_name(name: str) -> bool:
    return isinstance(name, str) and name.endswith(CREDENTIAL_SUFFIXES)


class CredentialStore:
    """A private snapshot of the startup environment.

    Credential-shaped variables are removed from the source environment when
    the store is captured.  The complete snapshot is retained because
    models.dev API templates also reference non-secret account and endpoint
    variables.
    """

    def __init__(self, values: dict[str, str]):
        self.__values = dict(values)

    @classmethod
    def capture(cls, environ: MutableMapping[str, str]):
        values = dict(environ)
        for name in list(environ):
            if is_credential_name(name):
                environ.pop(name, None)
        return cls(values)

    def get(self, name: str, default: str = "") -> str:
        value = self.__values.get(name, default)
        return value if isinstance(value, str) else default

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    def first_available_name(self, names: Iterable[str]) -> str | None:
        for name in names:
            if self.has(name):
                return name
        return None

    def sanitized_environment(self) -> dict[str, str]:
        """Return captured startup configuration without known credentials."""
        return {
            name: value for name, value in self.__values.items()
            if not is_credential_name(name)
        }

    def credential_refs(self) -> frozenset[CredentialRef]:
        from .authentications import CredentialRef

        return frozenset(
            CredentialRef.environment(name)
            for name, value in self.__values.items()
            if is_credential_name(name) and value
        )

    def has_ref(self, credential: CredentialRef) -> bool:
        return (
            credential.kind == "env"
            and self.has(credential.name))

    def install_static_credentials(self, broker: CredentialBroker) -> None:
        """Register environment-shaped request secrets with the root broker."""
        from .authentications import CredentialRef

        for name, value in self.__values.items():
            if is_credential_name(name) and value:
                broker.install_static(
                    CredentialRef.environment(name), value)

    def __repr__(self) -> str:
        present = sum(1 for value in self.__values.values() if value)
        return f"CredentialStore(<redacted>, {present} populated names)"


class CredentialInventory:
    """Non-secret configuration and credential availability for a process."""

    def __init__(
            self, values: dict[str, str],
            credentials: Iterable[CredentialRef] = ()):
        self.__values = {
            name: value for name, value in values.items()
            if not is_credential_name(name)
        }
        self.__credentials = frozenset(credentials)

    @classmethod
    def from_environment(
            cls, environ: MutableMapping[str, str],
            credentials: Iterable[CredentialRef] = ()):
        return cls(dict(environ), credentials)

    def get(self, name: str, default: str = "") -> str:
        value = self.__values.get(name, default)
        return value if isinstance(value, str) else default

    def has(self, name: str) -> bool:
        if is_credential_name(name):
            return any(
                credential.kind == "env" and credential.name == name
                for credential in self.__credentials)
        return bool(self.get(name))

    def first_available_name(
            self, names: Iterable[str]) -> str | None:
        for name in names:
            if self.has(name):
                return name
        return None

    def credential_refs(self) -> frozenset[CredentialRef]:
        return self.__credentials

    def has_ref(self, credential: CredentialRef) -> bool:
        return credential in self.__credentials

    def __repr__(self) -> str:
        return (
            "CredentialInventory("
            f"{len(self.__values)} settings, "
            f"{len(self.__credentials)} credential names)")


class CredentialScrubError(RuntimeError):
    """The native startup environment could not be scrubbed safely."""


_PROCESS_CREDENTIALS: CredentialStore | None = None


def _native_environ(libc: ctypes.CDLL):
    """Return the platform's live ``char **environ`` pointer."""
    environ_type = ctypes.POINTER(ctypes.c_void_p)
    try:
        if sys.platform == "darwin":
            # Darwin documents _NSGetEnviron() in environ(7) for programs
            # which cannot safely refer to the environ symbol directly.
            get_environ = libc._NSGetEnviron
            get_environ.argtypes = ()
            get_environ.restype = ctypes.POINTER(environ_type)
            environ_pointer = get_environ()
            if not environ_pointer or not environ_pointer[0]:
                raise CredentialScrubError(
                    "_NSGetEnviron() returned no process environment"
                )
            return environ_pointer[0]
        return environ_type.in_dll(libc, "environ")
    except (AttributeError, OSError, ValueError) as error:
        raise CredentialScrubError(
            f"could not access the native process environment: {error}"
        ) from error


def _darwin_main_stack_bounds(libc: ctypes.CDLL) -> tuple[int, int]:
    """Return the original main-thread stack's low and high addresses."""
    try:
        pthread_main_np = libc.pthread_main_np
        pthread_main_np.argtypes = ()
        pthread_main_np.restype = ctypes.c_int
        pthread_self = libc.pthread_self
        pthread_self.argtypes = ()
        pthread_self.restype = ctypes.c_void_p
        pthread_get_stackaddr_np = libc.pthread_get_stackaddr_np
        pthread_get_stackaddr_np.argtypes = (ctypes.c_void_p,)
        pthread_get_stackaddr_np.restype = ctypes.c_void_p
        pthread_get_stacksize_np = libc.pthread_get_stacksize_np
        pthread_get_stacksize_np.argtypes = (ctypes.c_void_p,)
        pthread_get_stacksize_np.restype = ctypes.c_size_t
    except (AttributeError, OSError, ValueError) as error:
        raise CredentialScrubError(
            f"could not inspect the Darwin main-thread stack: {error}"
        ) from error

    if pthread_main_np() != 1:
        raise CredentialScrubError(
            "Darwin credential capture must run on the main thread"
        )
    thread = pthread_self()
    stack_high = pthread_get_stackaddr_np(thread)
    stack_size = int(pthread_get_stacksize_np(thread))
    if not thread or not stack_high or stack_size <= 0:
        raise CredentialScrubError(
            "Darwin returned invalid main-thread stack bounds"
        )
    stack_high = int(stack_high)
    stack_low = stack_high - stack_size
    if stack_low < 0:
        raise CredentialScrubError(
            "Darwin returned invalid main-thread stack bounds"
        )
    return stack_low, stack_high


def _validate_entries_in_range(
        matches: dict[str, list[tuple[int, int, int]]],
        low: int,
        high: int,
) -> None:
    """Require every complete native string to lie within one address range."""
    for name, entries in matches.items():
        for address, length, _prefix_length in entries:
            end = address + length + 1
            if address < low or end > high or end <= address:
                raise CredentialScrubError(
                    f"credential variable {name} was not stored in the "
                    "Darwin initial main-thread stack"
                )


def _native_credential_entries(
        names: list[str],
) -> dict[str, list[tuple[int, int, int]]]:
    """Preflight native ``environ`` entries without copying secret values."""
    try:
        libc = ctypes.CDLL(None)
        environ = _native_environ(libc)
        strlen = libc.strlen
        strlen.argtypes = (ctypes.c_void_p,)
        strlen.restype = ctypes.c_size_t
    except CredentialScrubError:
        raise
    except (AttributeError, OSError, ValueError) as error:
        raise CredentialScrubError(
            f"could not access the native process environment: {error}"
        ) from error

    prefixes = {
        name: os.fsencode(name) + b"="
        for name in names
    }
    matches: dict[str, list[tuple[int, int, int]]] = {
        name: [] for name in names
    }
    index = 0
    while True:
        address = environ[index]
        if not address:
            break
        length = int(strlen(address))
        for name, prefix in prefixes.items():
            prefix_length = len(prefix)
            if (
                    length >= prefix_length
                    and ctypes.string_at(address, prefix_length) == prefix
            ):
                matches[name].append(
                    (address, length, prefix_length))
        index += 1

    missing = [name for name, entries in matches.items() if not entries]
    if missing:
        rendered = ", ".join(sorted(missing))
        raise CredentialScrubError(
            "credential variables were present in os.environ but absent "
            f"from native environ: {rendered}"
        )
    if sys.platform == "darwin":
        stack_low, stack_high = _darwin_main_stack_bounds(libc)
        _validate_entries_in_range(matches, stack_low, stack_high)
    return matches


def _scrub_native_credentials(
        environ: MutableMapping[str, str],
        names: list[str],
) -> None:
    """Remove credentials while preserving the initial record boundaries."""
    matches = _native_credential_entries(names)

    # First erase every value but retain NAME=.  libc unsetenv() must still
    # be able to recognize each entry, including duplicate entries supplied
    # directly to execve().
    for entries in matches.values():
        for address, length, prefix_length in entries:
            value_length = length - prefix_length
            if value_length:
                ctypes.memset(
                    address + prefix_length,
                    ord("x"),
                    value_length,
                )

    # MutableMapping deletion on os.environ calls libc unsetenv(), removing
    # the entry from the active pointer vector as well as Python's mapping.
    for name in names:
        del environ[name]

    # On Linux these addresses are initial exec storage; on Darwin the
    # preflight above proved that they are in the initial main-thread stack.
    # unsetenv() removed pointers to them but did not free or unmap the
    # strings.  Fill the retained NAME= bytes without touching the final NUL,
    # preserving the same-length, NUL-delimited startup records.
    for entries in matches.values():
        for address, _length, prefix_length in entries:
            ctypes.memset(address, ord("x"), prefix_length)


def capture_process_credentials() -> CredentialStore:
    """Capture and scrub credentials once, at process startup.

    Linux exposes the initial exec environment through /proc/PID/environ even
    after unsetenv(), and Darwin's KERN_PROCARGS2 exposes the corresponding
    initial stack region.  On both platforms, overwrite the original
    credential records in place before continuing.  Other platforms retain
    the existing os.environ/unsetenv behavior.
    """
    global _PROCESS_CREDENTIALS
    if _PROCESS_CREDENTIALS is not None:
        return _PROCESS_CREDENTIALS

    values = dict(os.environ)
    names = [name for name in os.environ if is_credential_name(name)]
    if (
            (sys.platform.startswith("linux") or sys.platform == "darwin")
            and names
    ):
        _scrub_native_credentials(os.environ, names)
        store = CredentialStore(values)
    else:
        store = CredentialStore.capture(os.environ)
    _PROCESS_CREDENTIALS = store
    return store
