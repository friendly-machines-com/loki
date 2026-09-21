"""Executable-owned logging configuration and safe diagnostic payloads."""

import json
import logging
import logging.config
import os
import sys


def configure_logging(contained: bool = False) -> bool:
    """Configure after security initialization, before session cwd changes.

    Relative filenames assume the invoker's cwd is still in effect. The
    credential-owning terminal supervisor and ACP front spawn runtimes without
    changing cwd; subagents inherit the runtime's process cwd and apply their
    logical shell cwd later. Keep these launch paths and this call ordering in
    sync. Pass the filename literally: expanding '~' or collapsing '..' can
    select a different executable configuration file.

    ``contained`` says this process runs inside the container, which must not
    load a configuration the invoker names.  It cannot be assumed able to read
    that file or to write the handlers' targets -- on Windows it may only touch
    its granted trees -- and any path it *can* reach is one the model may be
    able to rewrite.  Such a process logs to stderr instead, and says so.
    """
    try:
        path = os.environ.get("LOKI_LOG_CONFIG")
        if path is not None and contained:
            print(
                "Logging configuration ignored: LOKI_LOG_CONFIG is not loaded "
                "inside the contained runtime; logging to stderr.",
                file=sys.stderr)
            path = None
        if path is not None:
            logging.config.fileConfig(
                path, disable_existing_loggers=False, encoding="utf-8")
        else:
            logger = logging.getLogger("loki_agent")
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
            logger.disabled = False
            logger.propagate = False
            logger.setLevel(
                logging.DEBUG if os.environ.get("LOKI_TRACE") == "1"
                else logging.WARNING)
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                "%(levelname)s %(name)s[%(process)d]: %(message)s"))
            logger.addHandler(handler)
        return True
    except Exception as error:
        # Configuration is trusted executable input, but its error text may
        # still contain control characters. Do not dump configuration contents.
        print(f"Logging configuration error: {ascii(str(error))}",
              file=sys.stderr)
        return False


def debug_json(logger, label, value):
    if logger.isEnabledFor(logging.DEBUG):
        sys.stdout.flush()
        logger.debug("%s\n%s", label, json.dumps(
            value, ensure_ascii=True, sort_keys=True, default=str))
