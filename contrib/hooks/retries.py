"""Exponential backoff for long-running Loki turn_error recovery.

A configured user turn uses this policy INSTEAD OF the fixed LLM request retry
policy. Loki supplies one retry_count shared by prepared-request replay and
history continuation; it retains request identity and owns cancellable waits.
By default, every formal turn_error remains retryable, including connection/DNS
failures before generation starts. Only --max-retries imposes a count limit.
Overload diagnostics select a slower initial delay using the same turn-wide
count, not a second budget. This script neither imports Loki nor retains state
or locates any other file. Provider retry-delay minimums are enforced by Loki,
not duplicated here.
"""

import argparse
import json
import math
import sys


def nonnegative_number(value):
    number = float(value)
    if number < 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def main():
    # argparse's default derives a label from the selected script pathname.
    # Help text needs a display label, not another interpretation of that key.
    parser = argparse.ArgumentParser(prog="retry policy", description=__doc__)
    parser.add_argument("--initial-delay", type=nonnegative_number, default=5.0)
    parser.add_argument("--overload-initial-delay", type=nonnegative_number, default=600.0)
    parser.add_argument("--multiplier", type=nonnegative_number, default=2.0)
    parser.add_argument("--max-delay", type=nonnegative_number, default=1800.0)
    parser.add_argument("--max-retries", type=int,
                        help="optional retry limit; by default, keep retrying")
    args = parser.parse_args()
    if args.multiplier < 1:
        parser.error("--multiplier must be at least 1")
    if args.max_retries is not None and args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    try:
        payload = json.load(sys.stdin)
    except (ValueError, UnicodeError) as error:
        parser.error(str(error))
    if not isinstance(payload, dict):
        parser.error("hook input must be an object")
    count = payload.get("retry_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        parser.error("retry_count must be a nonnegative integer")
    result = {"action": "finish"}
    if payload.get("event") == "turn_error" and (
            args.max_retries is None or count < args.max_retries):
        delay = args.initial_delay
        message = payload.get("error_message")
        # Loki already selected a formal failure. Its diagnostic only chooses
        # the backoff base; assistant text is not used to discover errors.
        if isinstance(message, str) and "overload" in message.casefold():
            delay = args.overload_initial_delay
        if delay:
            try:
                delay *= args.multiplier ** count
            except OverflowError:
                delay = args.max_delay
        result = {"action": "retry", "delay_seconds": min(delay, args.max_delay)}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
