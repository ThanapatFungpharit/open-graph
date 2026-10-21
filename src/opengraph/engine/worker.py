"""A child process that runs symbolic operations: ``python -m opengraph.engine.worker``.

SymPy is pure Python, so running it on a thread of the GUI process still holds the GIL
for long stretches and makes the interface stutter; a separate process doesn't, and can
be killed to cancel a computation that would never finish.

Protocol: length-prefixed pickled frames. The parent writes ``(task id, operation,
SymbolicTarget)`` to stdin; the worker answers each with ``(task id, ok, payload)`` on
stdout, where ``payload`` is the operation's result when ``ok`` and an error message
otherwise. Frames only ever travel between OpenGraph and its own child, so pickle is
safe here. Free of interface code, like the rest of the engine.
"""

from __future__ import annotations

import logging
import pickle
import struct
import sys
from collections.abc import Callable
from typing import Any, BinaryIO

from opengraph.engine import symbolic
from opengraph.engine.compiler import CasRequest, SymbolicTarget
from opengraph.engine.errors import MathError

log = logging.getLogger(__name__)

_HEADER = struct.Struct("<I")

OPERATIONS: dict[str, Callable[[Any], object]] = {
    "derivative": symbolic.derivative,
    "integral": symbolic.integral,
    "simplify": symbolic.simplify,
    "solve": symbolic.solve,
    "exact": symbolic.exact_value,
    "rearrange": symbolic.rearrange,
    "cas": symbolic.cas,
}


def encode(message: object) -> bytes:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    return _HEADER.pack(len(data)) + data


def decode(buffer: bytearray) -> list[object]:
    """Remove and return every complete frame at the start of ``buffer``."""
    messages = []
    while len(buffer) >= _HEADER.size:
        (length,) = _HEADER.unpack_from(buffer)
        end = _HEADER.size + length
        if len(buffer) < end:
            break
        messages.append(pickle.loads(bytes(buffer[_HEADER.size : end])))
        del buffer[:end]
    return messages


def run(operation: str, target: SymbolicTarget | CasRequest) -> tuple[bool, object]:
    """Run one operation: ``(True, result)`` or ``(False, error message)``."""
    try:
        return True, OPERATIONS[operation](target)
    except MathError as exc:
        return False, exc.message
    except Exception as exc:  # SymPy can fail in many ways; report, don't crash
        log.exception("Symbolic %s failed", operation)
        return False, f"SymPy could not complete this ({type(exc).__name__})"


def _read_exactly(stream: BinaryIO, size: int) -> bytes | None:
    data = b""
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def serve(requests: BinaryIO, replies: BinaryIO) -> None:
    """Answer requests until the parent closes the pipe."""
    while True:
        header = _read_exactly(requests, _HEADER.size)
        if header is None:
            return
        body = _read_exactly(requests, _HEADER.unpack(header)[0])
        if body is None:
            return
        task_id, operation, target = pickle.loads(body)
        ok, payload = run(operation, target)
        replies.write(encode((task_id, ok, payload)))
        replies.flush()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    replies = sys.stdout.buffer
    sys.stdout = sys.stderr  # stray prints must not corrupt the reply stream
    serve(sys.stdin.buffer, replies)


if __name__ == "__main__":
    main()
