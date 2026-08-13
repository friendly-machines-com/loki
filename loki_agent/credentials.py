"""In-memory startup credentials with child-environment scrubbing."""

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

    def __repr__(self) -> str:
        present = sum(1 for value in self.__values.values() if value)
        return f"CredentialStore(<redacted>, {present} populated names)"
