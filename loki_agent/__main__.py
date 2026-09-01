import sys

from .credentials import capture_process_credentials
from .process_protections import (
    ProcessProtectionError,
    protect_credential_process,
)


def main() -> int:
    # Scrub the initial exec environment before importing the terminal
    # frontend and the rest of Loki. A delegated subagent already owns its
    # capability descriptor at exec time, so establish the ptrace/proc
    # boundary here too, before importing the much larger runtime.
    credentials = capture_process_credentials()
    try:
        protect_credential_process()
    except ProcessProtectionError as error:
        print(f"Security initialization error: {error}", file=sys.stderr)
        return 2
    if sys.argv[1:2] == ["--subagent"]:
        from .subagents import main as subagent_main
        return subagent_main(sys.argv[2:], credentials)
    from .terminal_frontend import main as terminal_main
    return terminal_main(credentials)


if __name__ == "__main__":
    raise SystemExit(main())
