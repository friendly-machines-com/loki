"""Public invocation and trusted tool policy launch arguments."""

import getopt
import sys


_ALLOW_ABSTRACT_UNIX = '--allow-tool-host-abstract-unix'
_ABSTRACT_UNIX_POLICY = '--tool-host-abstract-unix'


USAGE = f"""\
usage: loki [options]

Options:
  -r, --resume LOG        resume a saved chat log (bare --resume: picker)
  -p, --prompt TEXT       one prompt for --headless mode
      --headless          headless single-prompt mode
      --toolset NAME      toolset for headless mode
      --shell-cwd PATH    working directory for tools and Bash
                          (an explicit value outranks a resumed one)
      --dangerously-skip-permissions
                          skip permission prompts
      {_ALLOW_ABSTRACT_UNIX}
                          allow tools to reach host abstract Unix sockets
                          (Linux, must be the first argument)
      --print-default-system-prompt
                          print built-in instructions and exit (use alone)
  -h, --help              show this help and exit

Without options, loki starts the interactive TUI.
Use /status for the current connection, /status all for all known connections.
Add --json for JSON; /status save saves this runtime's response observations.
Use loki status [--json] [--endpoint URL] to inspect saved response headers.
Use /account for live provider usage and limit resets, when supported.
Use /queue to inspect queued prompts and staged images while a turn runs.
"""


CLI_SHORT_OPTS = 'r:p:h'


CLI_LONG_OPTS = ['resume=', 'prompt=', 'headless', 'toolset=',
                 'shell-cwd=',
                 'dangerously-skip-permissions', 'help']


def public_abstract_unix_permission(args: list[str]) -> tuple[bool, list[str]]:
    """Only an exact leading option grants access; leave all other args alone."""
    if sys.platform.startswith('linux') and args[:1] == [_ALLOW_ABSTRACT_UNIX]:
        return True, args[1:]
    return False, args


def abstract_unix_launch_args(allowed: bool) -> list[str]:
    # Use existing private argv, not an environment setting or conversation
    # option. Send deny explicitly too: a pre-policy receiver can reject the
    # new launch grammar rather than start the command without the policy.
    if not sys.platform.startswith('linux'):
        return []
    return [_ABSTRACT_UNIX_POLICY, 'allow' if allowed else 'deny']


def read_abstract_unix_launch_args(args: list[str]) -> tuple[bool, list[str]]:
    # Called only at the designated private argument position, never on
    # target argv or prompts. A live older parent still sending the original
    # layout lends no permission to a newly execed bootstrap.
    if not sys.platform.startswith('linux') or args[:1] != [_ABSTRACT_UNIX_POLICY]:
        return False, args
    if len(args) < 2 or args[1] not in ['allow', 'deny']:
        raise ValueError('invalid permission for host abstract Unix sockets')
    return args[1] == 'allow', args[2:]


def parse_cli_args(args):
    """Normalize and getopt-parse CLI args (raises getopt.GetoptError)."""
    # getopt's "resume=" requires a value; normalize a bare `--resume` to
    # `--resume=` so it opens the picker instead of erroring out.
    args = ['--resume=' if a == '--resume' else a for a in args]
    return getopt.getopt(args, CLI_SHORT_OPTS, CLI_LONG_OPTS)


def preflight(args, *, reject_positionals=False):
    """Handle help/argument errors before storage or terminal initialization.

    Windows historically rejects positional arguments at its launch gate;
    ordinary terminal parsing leaves them to its existing runtime behavior.
    Both paths use the same option grammar and safe diagnostic rendering.
    """
    try:
        options, positional = parse_cli_args(args)
        if reject_positionals and positional:
            raise getopt.GetoptError("unexpected positional arguments")
    except getopt.GetoptError as error:
        print(f"loki: {str(error)!r}", file=sys.stderr)
        print(USAGE, end="", file=sys.stderr)
        return 2
    if any(name in ["-h", "--help"] for name, _value in options):
        print(USAGE, end="")
        return 0
    return None
