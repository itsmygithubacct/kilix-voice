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
import re

from . import models

OP_SPEAK = "speak"
OP_STOP_SPEECH = "stop-speech"
OP_DICTATE = "dictate"
OP_STOP_DICTATION = "stop-dictation"
OP_STATUS = "status"

OPS = (OP_SPEAK, OP_STOP_SPEECH, OP_DICTATE, OP_STOP_DICTATION, OP_STATUS)

MAX_ID_CHARS = 64
# AF_UNIX/SOCK_SEQPACKET has a platform message ceiling below the daemon's old
# one-megabyte read guard.  This conservative bound is shared by clients and
# the daemon so an oversized request is rejected explicitly instead of failing
# at send(2) with an opaque EMSGSIZE.
MAX_REQUEST_BYTES = 192 * 1024

# A caller may bound how long it is willing to wait.  Relative milliseconds,
# not an absolute instant: the two ends do not share a clock, and a monotonic
# offset cannot be invalidated by a wall-clock step.  The ceiling keeps a typo
# from parking a job for a day.
MAX_DEADLINE_MS = 24 * 60 * 60 * 1000

# Errors cross the boundary as a CODE from this closed set plus prose.  The
# prose is for a human and may change; the code is the contract and may not.
# Without a code every caller ends up matching on message text, which makes the
# text load-bearing and unfixable -- and invites internals onto the wire.
ERR_MALFORMED = "malformed"
ERR_UNSUPPORTED = "unsupported"
ERR_BUSY = "busy"
ERR_DEADLINE = "deadline"
ERR_CANCELLED = "cancelled"
ERR_NOT_FOUND = "not-found"
ERR_DENIED = "denied"
ERR_TOO_LARGE = "too-large"
ERR_UNAVAILABLE = "unavailable"
ERR_INTERNAL = "internal"
ERROR_CODES = (ERR_MALFORMED, ERR_UNSUPPORTED, ERR_BUSY, ERR_DEADLINE,
               ERR_CANCELLED, ERR_NOT_FOUND, ERR_DENIED, ERR_TOO_LARGE,
               ERR_UNAVAILABLE, ERR_INTERNAL)

TTS_RATE_CHOICES = (120, 150, 170, 200, 240)
_VOICE_TOKEN = re.compile(r"^[A-Za-z0-9_+-]{1,32}$")


class ProtocolError(ValueError):
    """A malformed or unsafe message; the text says what to send instead."""


class MessageTooLarge(ProtocolError):
    """The frame exceeds MAX_REQUEST_BYTES.

    A distinct type so a caller can substitute its own domain wording without
    matching on message text -- which is exactly the coupling the error-code
    vocabulary below exists to remove.
    """


def encode(msg: dict) -> bytes:
    """Return one UTF-8 line, newline-terminated, for ``msg``."""
    if not isinstance(msg, dict):
        raise ProtocolError(
            f"a protocol message must be a dict, got {type(msg).__name__}. "
            "Wrap the value, for example {'op': 'status'}.")
    try:
        line = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
        frame = line.encode("utf-8") + b"\n"
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ProtocolError(
            f"message is not JSON-serialisable ({error}). Use only str, int, "
            "float, bool, None, list and dict values.") from error
    # Size is checked OUTSIDE the try: ProtocolError is a ValueError, so raising
    # it in there would be caught by this very handler and re-reported as
    # "not JSON-serialisable" -- a wrong diagnosis for a correct message that is
    # merely too big. The existing suite caught exactly that.
    if len(frame) > MAX_REQUEST_BYTES:
        raise MessageTooLarge(
            f"message is {len(frame)} bytes; the limit is {MAX_REQUEST_BYTES}. "
            "Send the payload as a file path or split it across messages.")
    return frame


def decode(raw: bytes | str) -> dict:
    """Return the object encoded in one line; raise ProtocolError otherwise."""
    # Size is checked BEFORE decode and before json.loads, so an oversized
    # frame costs no UTF-8 pass and no parser allocation.
    measured = None
    if isinstance(raw, str):
        measured = len(raw.encode("utf-8"))
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        measured = len(raw)
    # Anything else falls through to the type handling below, which raises a
    # ProtocolError naming the bad type. Measuring it here would raise TypeError
    # instead -- the existing suite caught that.
    if measured is not None and measured > MAX_REQUEST_BYTES:
        raise MessageTooLarge(
            f"message is {measured} bytes; the limit is {MAX_REQUEST_BYTES}. "
            "Refused before decoding.")
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


def _deadline_ms(raw: object) -> int:
    """Return a positive relative deadline in milliseconds, or raise."""
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ProtocolError(
            f"'deadline_ms' must be an integer number of milliseconds, got "
            f"{type(raw).__name__}. Send for example 5000 for five seconds.")
    if raw <= 0:
        raise ProtocolError(
            f"'deadline_ms' must be greater than zero, got {raw}. A deadline "
            "that has already passed cannot be met; omit it to wait "
            "indefinitely.")
    if raw > MAX_DEADLINE_MS:
        raise ProtocolError(
            f"'deadline_ms' is {raw}; the ceiling is {MAX_DEADLINE_MS} "
            "(24 hours). Use a shorter bound.")
    return raw


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
    if "deadline_ms" in msg:
        request["deadline_ms"] = _deadline_ms(msg.get("deadline_ms"))
    if op == OP_SPEAK:
        text = msg.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ProtocolError(
                'speak requires a non-empty "text" string. Send '
                '{"op":"speak","text":"…"}; the caller decides what the '
                "extent setting means before it sends.")
        request["text"] = text
        if "model" in msg:
            model = msg.get("model")
            if not isinstance(model, str) or model not in models.TTS_MODEL_IDS:
                raise ProtocolError(
                    f"'model' must be one of: {', '.join(models.TTS_MODEL_IDS)}. "
                    "It selects a registered local synthesizer; executable "
                    "paths and download URLs are never accepted.")
            request["model"] = model
        if "voice" in msg:
            voice = msg.get("voice")
            if not isinstance(voice, str) or not _VOICE_TOKEN.fullmatch(voice):
                raise ProtocolError(
                    "'voice' must be 1-32 characters from [A-Za-z0-9_+-], "
                    "such as en-us or us1.")
            request["voice"] = voice
        if "rate" in msg:
            rate = msg.get("rate")
            if (not isinstance(rate, int) or isinstance(rate, bool)
                    or rate not in TTS_RATE_CHOICES):
                raise ProtocolError(
                    f"'rate' must be one of: "
                    f"{', '.join(map(str, TTS_RATE_CHOICES))} words per minute.")
            request["rate"] = rate
    elif op == OP_DICTATE:
        request["sock"] = _validated_socket(msg.get("sock"), session_dir)
    return request


def reply_ok(request_id: str = "", **fields: object) -> dict:
    """Return a success reply, optionally carrying status fields."""
    reply: dict = {"ok": True, "id": request_id}
    reply.update(fields)
    return reply


def reply_error(message: str, code: str = ERR_INTERNAL) -> dict:
    """Return a failure reply carrying a code from the closed set.

    ``message`` is prose for a human and may be reworded freely; ``code`` is the
    contract a caller may branch on.  An unknown code is refused rather than
    forwarded, so the vocabulary cannot drift open one caller at a time.
    """
    if code not in ERROR_CODES:
        raise ProtocolError(
            f"unknown error code {code!r}. Use one of: "
            f"{', '.join(ERROR_CODES)}.")
    return {"ok": False, "error": message, "code": code}


def dictation_partial(text: str) -> dict:
    """Return an in-progress recognition datagram."""
    return {"partial": text}


def dictation_final(text: str) -> dict:
    """Return the final recognition datagram for one turn."""
    return {"final": text}


def dictation_error(message: str) -> dict:
    """Return a dictation failure datagram."""
    return {"error": message}
