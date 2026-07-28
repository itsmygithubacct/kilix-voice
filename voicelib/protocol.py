"""Wire format for the control socket and the dictation datagrams.

Line-delimited JSON: one object per line, one message per send.  JSON escapes
every newline inside a string, so a message can never span lines however odd
the text being spoken is.

This module is also where a request stops being caller-supplied data and starts
being something the daemon acts on — see ``validate_request``.
"""

from __future__ import annotations

import json
import os

OP_SPEAK = "speak"
OP_STOP_SPEECH = "stop-speech"
OP_DICTATE = "dictate"
OP_STOP_DICTATION = "stop-dictation"
OP_STATUS = "status"

OPS = (OP_SPEAK, OP_STOP_SPEECH, OP_DICTATE, OP_STOP_DICTATION, OP_STATUS)

MAX_ID_CHARS = 64


class ProtocolError(ValueError):
    """A malformed or unsafe message; the text says what to send instead."""


def encode(msg: dict) -> bytes:
    """Return one UTF-8 line, newline-terminated, for ``msg``."""
    if not isinstance(msg, dict):
        raise ProtocolError(
            f"a protocol message must be a dict, got {type(msg).__name__}. "
            "Wrap the value, for example {'op': 'status'}.")
    try:
        line = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ProtocolError(
            f"message is not JSON-serialisable ({error}). Use only str, int, "
            "float, bool, None, list and dict values.") from error
    return line.encode("utf-8") + b"\n"


def decode(raw: bytes | str) -> dict:
    """Return the object encoded in one line; raise ProtocolError otherwise."""
    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = bytes(raw).decode("utf-8")
        except (TypeError, UnicodeDecodeError) as error:
            raise ProtocolError(
                f"message is not valid UTF-8 ({error}). Send text encoded as "
                "UTF-8; kilix-voice speaks no other encoding.") from error
    text = text.strip()
    if not text:
        raise ProtocolError(
            "empty message. Send one JSON object per line, for example "
            '{"op":"status"}.')
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProtocolError(
            f"invalid JSON at column {error.colno}: {error.msg}. Send one "
            'JSON object per line, for example {"op":"status"}.') from error
    if not isinstance(msg, dict):
        raise ProtocolError(
            f"expected a JSON object, got {type(msg).__name__}. Send a "
            'mapping such as {"op":"speak","text":"hello"}.')
    return msg


def _request_id(raw: object) -> str:
    """Return a request id echoed back on the reply; "" when unset."""
    if raw is None:
        return ""
    if not isinstance(raw, (str, int)) or isinstance(raw, bool):
        raise ProtocolError(
            f"'id' must be a string or integer, got {type(raw).__name__}. "
            "Omit it to let the reply carry an empty id.")
    text = str(raw)
    if len(text) > MAX_ID_CHARS:
        raise ProtocolError(
            f"'id' is {len(text)} characters; the limit is {MAX_ID_CHARS}. "
            "Use a short correlation token such as a counter.")
    return text


def _validated_socket(raw: object, session_dir: str) -> str:
    """Resolve a caller-supplied dictation socket inside the session directory.

    This is a security boundary, not a tidiness check.  The daemon runs as the
    user and will connect to whatever path a request names, so an unchecked
    'sock' turns any request into "deliver everything the microphone hears to
    this path".  Both sides are therefore realpath()ed before the containment
    test: realpath resolves symlinks in every component including the last, so
    a link planted inside the session directory cannot point the daemon out of
    it.  The resolved path is what the caller gets back, so the daemon connects
    to exactly what was approved instead of re-resolving the original string.

    Containment is the perimeter, not the whole defence: the session directory
    is mode 0700 and the daemon checks SO_PEERCRED on every accept, so an
    attacker who could plant a link in there is already the user.
    """
    if not isinstance(raw, str) or not raw:
        raise ProtocolError(
            "dictate requires a 'sock' path. The kitty fork creates it as "
            "<session>/dictate-<pane>.sock; pass that path.")
    if "\x00" in raw:
        raise ProtocolError(
            "'sock' must not contain NUL bytes. Pass the plain socket path.")
    if not os.path.isabs(raw):
        # A relative path would be resolved against the daemon's cwd, which the
        # sender does not know; refuse rather than guess what it meant.
        raise ProtocolError(
            f"'sock' must be an absolute path, got {raw!r}. Pass the full "
            "<session>/dictate-<pane>.sock path.")
    root = os.path.realpath(os.path.expanduser(str(session_dir)))
    target = os.path.realpath(raw)
    try:
        contained = os.path.commonpath((root, target)) == root
    except ValueError as error:
        # Mixed absolute/relative roots only; both are absolute here, so this
        # means the session directory itself was passed in unusable.
        raise ProtocolError(
            f"cannot compare {raw!r} against the session directory "
            f"{session_dir!r}: {error}.") from error
    if not contained or target == root:
        raise ProtocolError(
            f"refusing 'sock' outside the session directory: {raw!r} resolves "
            f"to {target!r}, which is not a path inside {root!r}. Dictation "
            "sockets are created by the kitty fork as "
            "<session>/dictate-<pane>.sock.")
    return target


def validate_request(msg: dict, session_dir: str) -> dict:
    """Return a normalised copy of a control request, or raise ProtocolError.

    Only the fields an op actually uses survive, so nothing a caller invents can
    reach the daemon's dispatch as a surprise keyword.
    """
    if not isinstance(msg, dict):
        raise ProtocolError(
            f"a request must be a JSON object, got {type(msg).__name__}. "
            'Send for example {"op":"status"}.')
    op = msg.get("op")
    if not isinstance(op, str) or op not in OPS:
        raise ProtocolError(
            f"unknown op {op!r}. Use one of: {', '.join(OPS)}.")
    request: dict = {"op": op, "id": _request_id(msg.get("id"))}
    if op == OP_SPEAK:
        text = msg.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ProtocolError(
                'speak requires a non-empty "text" string. Send '
                '{"op":"speak","text":"…"}; the caller decides what the '
                "extent setting means before it sends.")
        request["text"] = text
    elif op == OP_DICTATE:
        request["sock"] = _validated_socket(msg.get("sock"), session_dir)
    return request


def reply_ok(request_id: str = "", **fields: object) -> dict:
    """Return a success reply, optionally carrying status fields."""
    reply: dict = {"ok": True, "id": request_id}
    reply.update(fields)
    return reply


def reply_error(message: str) -> dict:
    """Return a failure reply. ``message`` must say what the user should do."""
    return {"ok": False, "error": message}


def dictation_partial(text: str) -> dict:
    """Return an in-progress recognition datagram."""
    return {"partial": text}


def dictation_final(text: str) -> dict:
    """Return the final recognition datagram for one turn."""
    return {"final": text}


def dictation_error(message: str) -> dict:
    """Return a dictation failure datagram."""
    return {"error": message}
