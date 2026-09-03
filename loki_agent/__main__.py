"""Terminal/headless supervisor and its internal runtime dispatch.

The public invocation captures credentials and supervises a newly execed copy
of the same sanctioned executable. Internal ``--runtime`` and ``--subagent``
invocations consume delegated descriptors; they never capture credentials or
construct root authentication authority.
"""

import asyncio
import os
import sys

from .credentials import capture_process_credentials
from .runtime_isolations import (
    RuntimeIsolationError,
    isolate_credential_directory,
)
from .process_protections import (
    ProcessProtectionError,
    protect_credential_process,
)


def _descriptor(value: str, description: str) -> int:
    try:
        fd = int(value)
        if fd < 3:
            raise ValueError()
        os.fstat(fd)
        return fd
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid {description} descriptor") from error


def _terminal_runtime_arguments(args):
    if (
            len(args) < 5
            or args[0] != "--session-owner-fd"
            or args[2] != "--credential-capability-fd"
            or args[4] != "--"
    ):
        raise ValueError("invalid internal terminal runtime arguments")
    return (
        _descriptor(args[1], "session owner"),
        _descriptor(args[3], "credential capability"),
        args[5:],
    )


def _protect_runtime() -> bool:
    # Isolation happens before the large runtime import and while this newly
    # execed Python process is still single-threaded.
    isolate_credential_directory()
    return protect_credential_process()


def _report_security_error(error) -> int:
    print(f"Security initialization error: {error}", file=sys.stderr)
    return 2


def main() -> int:
    if sys.argv[1:2] == ["--runtime"]:
        try:
            owner_fd, capability_fd, args = (
                _terminal_runtime_arguments(sys.argv[2:]))
            _protect_runtime()
        except (ProcessProtectionError, RuntimeIsolationError,
                ValueError) as error:
            return _report_security_error(error)
        from .terminal_frontend import main as terminal_main
        return terminal_main(args, owner_fd, capability_fd)

    if sys.argv[1:2] == ["--subagent"]:
        try:
            # A subagent can only be delegated by the already-isolated
            # terminal runtime, so it inherits the credential cover mount.
            # Re-unsharing would add a namespace level for no security gain,
            # can hit nesting/policy limits, and would undermine the simple
            # invariant that only the first runtime establishes this view.
            protect_credential_process()
        except ProcessProtectionError as error:
            return _report_security_error(error)
        from .subagents import main as subagent_main
        return subagent_main(sys.argv[2:])

    # Public entrypoints alone capture startup credentials. The child runtime
    # is execed with this store's sanitized environment and therefore cannot
    # construct a parallel root authority from environment values.
    credentials = capture_process_credentials()
    try:
        protect_credential_process()
    except ProcessProtectionError as error:
        return _report_security_error(error)
    if sys.argv[1:2] == ["auth"]:
        from .authentication_commands import main as authentication_main
        return authentication_main(
            sys.argv[2:], program=sys.argv[0])
    from .credential_storages import (
        CredentialStorageError,
        JsonCredentialStorage,
    )
    from .credential_supervisors import CredentialSupervisor

    try:
        supervisor = CredentialSupervisor(
            credentials, JsonCredentialStorage())
    except CredentialStorageError as error:
        print(f"Credential storage error: {error}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(supervisor.run_terminal_runtime(
            sys.argv[0], sys.argv[1:]))
    except OSError as error:
        # sys.argv[0] is the exact sanctioned executable selected by the
        # caller (for example ./loki.py or an installed ``loki`` script).
        # Do not replace it with a random ambient Python interpreter.
        print(f"Could not start Loki runtime: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
