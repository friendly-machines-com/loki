"""Terminal/headless invocation grammar, independent of UI and execution."""

import getopt
import sys


USAGE = """\
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
