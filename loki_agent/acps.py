"""ACP transport: JSON-RPC over stdio, one message per line.

The front process speaks this on its real stdin/stdout; worker processes
speak it on socketpairs.  fd 1 carries protocol messages only, so the
front process quarantines it (see quarantine_stdout) and every other
writer in the process inherits a devnull instead.
"""

from __future__ import annotations

import json
import os
import sys


class TransportError(Exception):
    pass


def make_writer(fd: int):
    """Line-buffered writer for one JSON-RPC message per line.

    Payload and delimiter are separate writes: no concatenation means no
    second copy of a potentially large message, and the newline doubles
    as the flush marker.
    """
    stream = os.fdopen(os.dup(fd), "w", encoding="utf-8", buffering=1)

    def write(message: dict) -> None:
        stream.write(json.dumps(message, ensure_ascii=False))
        stream.write("\n")
        stream.flush()

    return write


def read_messages(fin):
    """Yield parsed JSON-RPC messages, one per line; stop at EOF."""
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise TransportError(f"line is not JSON: {error}") from error
        if not isinstance(message, dict):
            raise TransportError("line is not a JSON object")
        yield message


def response(request_id, result=None, error=None) -> dict:
    message = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    return message


def notification(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "method": method, "params": params}


def request(request_id, method: str, params: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


PARSE_ERROR = -32700
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def quarantine_stdout() -> None:
    """Reserve fd 1 for the protocol; everything else writes devnull.

    The original fd 1 is dup'd before being replaced, so the protocol
    writer keeps working; code that prints to stdout (ours or a
    library's) silently discards instead of corrupting the message
    stream.  Stderr stays untouched: it is ACP's log channel.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    # Replace the Python-level objects too, or print() would keep its old
    # buffer into the (now devnull) fd 1 while flush ordering gets strange.
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", buffering=1)
    sys.__stdout__ = sys.stdout
