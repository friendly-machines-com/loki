"""User-owned base instructions for new main conversations.

Runtime context and specialized agent instructions remain with their callers.
The resolved text is saved in the conversation, not cached or persisted here.
"""

import logging
import os

from . import paths


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful system agent running in a terminal. "
    "Prefer Glob/Grep/Read over Bash equivalents (find/grep/cat). "
    "Always Read a file before editing or overwriting it. "
    "Use TodoWrite to plan multi-step work. Keep responses concise.\n"
)


def load_system_prompt():
    """Read optional literal UTF-8 instructions without writing configuration.

    Empty files intentionally remove the base instructions. Missing files use
    the built-in text silently; other read errors are reported, never repaired.
    """
    try:
        file = os.path.join(paths.loki_config_dir(), "SYSTEM.md")
        with open(file, encoding="utf-8") as source:
            return source.read()
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT
    except (OSError, UnicodeError) as error:
        logging.getLogger(__name__).warning(
            "%s", ascii(f"Could not read SYSTEM.md: {error}. "
                        "Using the built-in system prompt for this conversation."))
        return DEFAULT_SYSTEM_PROMPT
