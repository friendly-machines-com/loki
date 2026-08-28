"""In-memory startup credentials with process-environment scrubbing."""

import ctypes
import os
import sys
from collections.abc import Iterable, MutableMapping


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

    def first_available(self, names: Iterable[str]) -> tuple[str | None, str]:
        for name in names:
            value = self.get(name)
            if value:
                return name, value
        return None, ""

    def startup_environment(self) -> dict[str, str]:
        """Return a fresh copy of the captured startup environment.

        This is intentionally credential-bearing.  It exists for the ACP
        front to bootstrap a worker, which immediately captures and scrubs
        its own process environment.
        """
        return dict(self.__values)

    def __repr__(self) -> str:
        present = sum(1 for value in self.__values.values() if value)
        return f"CredentialStore(<redacted>, {present} populated names)"


class CredentialScrubError(RuntimeError):
    """The native startup environment could not be scrubbed safely."""


_PROCESS_CREDENTIALS: CredentialStore | None = None


def _native_credential_entries(
        names: list[str],
) -> dict[str, list[tuple[int, int, int]]]:
    """Preflight native ``environ`` entries without copying secret values."""
    try:
        libc = ctypes.CDLL(None)
        environ = ctypes.POINTER(ctypes.c_void_p).in_dll(libc, "environ")
        strlen = libc.strlen
        strlen.argtypes = (ctypes.c_void_p,)
        strlen.restype = ctypes.c_size_t
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

    # These addresses are initial exec storage: unsetenv() removed pointers
    # to them but did not unmap the strings.  Fill the retained NAME= bytes
    # without touching the final NUL, so /proc/PID/environ remains a sequence
    # of same-length, NUL-delimited records.
    for entries in matches.values():
        for address, _length, prefix_length in entries:
            ctypes.memset(address, ord("x"), prefix_length)


def capture_process_credentials() -> CredentialStore:
    """Capture and scrub credentials once, at process startup.

    Linux exposes the initial exec environment through /proc/PID/environ even
    after unsetenv().  On Linux, overwrite those original credential records
    in place before continuing.  Other platforms retain the existing
    os.environ/unsetenv behavior.
    """
    global _PROCESS_CREDENTIALS
    if _PROCESS_CREDENTIALS is not None:
        return _PROCESS_CREDENTIALS

    values = dict(os.environ)
    names = [name for name in os.environ if is_credential_name(name)]
    if sys.platform.startswith("linux") and names:
        _scrub_native_credentials(os.environ, names)
        store = CredentialStore(values)
    else:
        store = CredentialStore.capture(os.environ)
    _PROCESS_CREDENTIALS = store
    return store
