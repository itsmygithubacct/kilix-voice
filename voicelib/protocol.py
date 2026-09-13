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

# Protocol identity. A caller MAY declare the wire contract it was written
# against; omitting it means "whatever the daemon speaks", which is what every
# existing client does and must keep working.
#
# Major is the compatibility boundary: same major = the request shapes this
# module validates are unchanged, so a newer minor may add optional fields and
# an older client still parses every reply it understands. A different major
# means a shape it validates has changed meaning, and guessing is worse than
# refusing -- the caller learns immediately instead of having a field silently
# reinterpreted.
PROTOCOL_SCHEMA = "kilix.voice.protocol/v1"
PROTOCOL_MAJOR = 1
PROTOCOL_MINOR = 1
PROTOCOL_VERSION = f"{PROTOCOL_MAJOR}.{PROTOCOL_MINOR}"
_VERSION_TOKEN = re.compile(r"^(\d{1,3})(?:\.(\d{1,3}))?$")

MAX_ID_CHARS = 64
# AF_UNIX/SOCK_SEQPACKET has a platform message ceiling below the daemon's old
# one-megabyte read guard.  This conservative bound is shared by clients and
# the daemon so an oversized request is rejected explicitly instead of failing
# at send(2) with an opaque EMSGSIZE.
MAX_REQUEST_BYTES = 192 * 1024

# Every reply and every dictation datagram fits the SMALLEST receive buffer a
# shipped client uses. kilix-tts, kilix-stt and the kitty fork read 1 << 16;
# kilix-avatar's speech.c reads 65535 and treats a full buffer as truncated.
# A reply bounded only by MAX_REQUEST_BYTES would still be cut off at every one
# of them and fail to decode there.
MAX_REPLY_BYTES = 64 * 1024 - 1

# Error prose is for a human. No message the daemon writes comes near this, so
# only text a caller inflated -- a quoted value, an exception carrying one --
# is ever cut.
MAX_ERROR_PROSE_CHARS = 4096
_TRUNCATED = " …[truncated]"

# A caller may bound how long it is willing to wait.  Relative milliseconds,
# not an absolute instant: the two ends do not share a clock, and a monotonic
# offset cannot be invalidated by a wall-clock step.  The ceiling keeps a typo
# from parking a job for a day.
MAX_DEADLINE_MS = 24 * 60 * 60 * 1000

# A06: a DECLARED audio byte limit that REFUSES. The capture path bounds a live
# stream by truncating it, which is right for a microphone and wrong for a
# request: silently handing back a prefix of what the caller asked about is a
# wrong answer wearing a success reply.
#
# The ceiling is INCLUSIVE: exactly this many bytes is accepted, one more is
# refused. An earlier comment here said "at or over this is refused", which
# contradicted the `>` in the code; the tests check both endpoints.
MAX_AUDIO_BYTES = 32 * 1024 * 1024

# A07: large audio is never embedded in JSON. A JSON message may carry a short
# preview or a descriptor, never a payload -- base64 in a control frame both
# blows the frame budget and forces the whole clip into memory to parse.
MAX_EMBEDDED_AUDIO_BYTES = 4 * 1024

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

# Linux PATH_MAX, counting the terminating NUL: a longer path cannot name a
# socket, and resolving one only produces a longer error.
_PATH_MAX = 4096
# Paths are quoted back at more length than other values: a refusal that cut
# the session directory off at 80 characters would hide the one thing a user
# needs to see, and PATH_MAX already bounds them.
_PATH_ECHO_CHARS = 512


class ProtocolError(ValueError):
    """A malformed or unsafe message; the text says what to send instead.

    The base carries NO code. A ProtocolError can also be raised inside a
    worker -- building a descriptor, say -- where the adapter's default of
    `unavailable` is the right reading; a code on the base would silently turn
    every one of those into `malformed`. The daemon's request-validation arm
    supplies `malformed` as ITS default instead.

    A raise site that knows the cause better passes ``code=`` for that one
    instance. It is an instance attribute rather than a subclass on purpose:
    the refusal of an incompatible protocol major is pinned by a frozen
    contract vector as exactly ProtocolError, so its code cannot come from a
    new type.
    """

    code: str | None = None

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            if code not in ERROR_CODES:
                raise ValueError(
                    f"unknown error code {code!r}. Use one of: "
                    f"{', '.join(ERROR_CODES)}.")
            self.code = code


class MessageTooLarge(ProtocolError):
    """The frame exceeds MAX_REQUEST_BYTES.

    A distinct type so a caller can substitute its own domain wording without
    matching on message text -- which is exactly the coupling the error-code
    vocabulary below exists to remove.
    """

    code = ERR_TOO_LARGE

    def __init__(self, message: str = "", *, size: int | None = None,
                 limit: int | None = None) -> None:
        super().__init__(message)
        self.size = size
        self.limit = limit


def _echo(value: object, limit: int = 80) -> str:
    """Return ``repr(value)`` cut to ``limit`` characters, for quoting in prose.

    A refusal quotes the caller's value back so it says what was wrong -- but
    the size of that quote must not be the caller's to choose. An op of 95000
    backslashes repr'd to 190000 characters, JSON-escaped again into a reply
    of 380 KB, and that reply's MessageTooLarge ended the daemon's serve loop.
    """
    if isinstance(value, str) and len(value) > limit:
        return f"{value[:limit]!r}…({len(value)} chars)"
    try:
        text = repr(value)
    except ValueError:          # an int past the interpreter's digit limit
        return f"<{type(value).__name__} too large to show>"
    return text if len(text) <= limit else f"{text[:limit]}…({len(text)} chars)"


def _cut_prose(message: object, limit: int) -> object:
    """Return error prose no longer than ``limit`` characters plus a marker."""
    if isinstance(message, str) and len(message) > limit:
        return message[:limit] + _TRUNCATED
    return message


def encode(msg: dict, *, limit: int = MAX_REQUEST_BYTES) -> bytes:
    """Return one UTF-8 line, newline-terminated, for ``msg``.

    ``limit`` is the largest frame accepted: MAX_REQUEST_BYTES for a request,
    MAX_REPLY_BYTES for anything the daemon sends to a client.
    """
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
    if len(frame) > limit:
        raise MessageTooLarge(
            f"message is {len(frame)} bytes; the limit is {limit}. "
            "Send the payload as a file path or split it across messages.",
            size=len(frame), limit=limit)
    return frame


def encode_reply(reply: dict, limit: int = MAX_REPLY_BYTES) -> bytes:
    """Return ONE frame of at most ``limit`` bytes for ``reply``, always.

    A request must never be able to leave the daemon holding a reply it cannot
    send, because every reply that failed to encode used to raise straight out
    of the connection handler. So: the reply as built; failing that, an error
    reply with its prose cut far enough that it must fit (6 bytes per
    character is JSON's worst case); failing that, a constant `internal` reply
    that names the failure's type. An over-size SUCCESS reply is never
    trimmed -- a partial status is a wrong answer wearing a success -- so it
    becomes that constant as well.
    """
    try:
        return encode(reply, limit=limit)
    except ProtocolError as error:
        failure = error
    if (isinstance(failure, MessageTooLarge) and isinstance(reply, dict)
            and reply.get("ok") is False and isinstance(reply.get("error"), str)):
        shorter = dict(reply)
        shorter["error"] = _cut_prose(reply["error"], MAX_ERROR_PROSE_CHARS // 2)
        try:
            return encode(shorter, limit=limit)
        except ProtocolError as error:
            failure = error
    return encode({
        "ok": False,
        "error": (f"kilix-voiced built a reply it could not send "
                  f"({type(failure).__name__}); this is a bug, and the daemon "
                  "is still running."),
        "code": ERR_INTERNAL,
    }, limit=limit)


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
            "Refused before decoding.", size=measured, limit=MAX_REQUEST_BYTES)
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
    except (ValueError, RecursionError) as error:
        # Not every parser refusal is a JSONDecodeError. An integer literal
        # past the interpreter's digit limit raises a plain ValueError, and
        # nesting deeper than the scanner's recursion limit raises
        # RecursionError. Neither is a ProtocolError, so each escaped the
        # daemon's validation arm and ended its serve loop.
        raise ProtocolError(
            f"invalid JSON ({type(error).__name__}): {_echo(str(error))}. "
            'Send one JSON object per line, for example {"op":"status"}.'
        ) from error
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
    # Both checked BEFORE the path reaches the filesystem layer. A lone
    # surrogate made os.path.realpath raise UnicodeEncodeError, which is not a
    # ProtocolError and ended the serve loop; a path longer than PATH_MAX
    # cannot name a socket and only grows the refusal that quotes it.
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ProtocolError(
            "'sock' is not a valid UTF-8 path: it contains a lone surrogate. "
            "Pass the socket path as UTF-8 text.") from error
    if len(encoded) >= _PATH_MAX:
        raise ProtocolError(
            f"'sock' is {len(encoded)} bytes, longer than PATH_MAX "
            f"({_PATH_MAX}). Pass the <session>/dictate-<pane>.sock path.")
    shown = _echo(raw, _PATH_ECHO_CHARS)
    if not os.path.isabs(raw):
        # A relative path would be resolved against the daemon's cwd, which the
        # sender does not know; refuse rather than guess what it meant.
        raise ProtocolError(
            f"'sock' must be an absolute path, got {shown}. Pass the full "
            "<session>/dictate-<pane>.sock path.")
    try:
        root = os.path.realpath(os.path.expanduser(str(session_dir)))
        target = os.path.realpath(raw)
        contained = os.path.commonpath((root, target)) == root
    except (ValueError, OSError) as error:
        # Mixed absolute/relative roots, or a path the filesystem layer will
        # not resolve. Both inputs are checked above, so this means the
        # session directory itself was passed in unusable.
        raise ProtocolError(
            f"cannot compare {shown} against the session directory "
            f"{_echo(session_dir, _PATH_ECHO_CHARS)}: "
            f"{_echo(str(error), _PATH_ECHO_CHARS)}.") from error
    if not contained or target == root:
        raise ProtocolError(
            f"refusing 'sock' outside the session directory: {shown} resolves "
            f"to {_echo(target, _PATH_ECHO_CHARS)}, which is not a path inside "
            f"{_echo(root, _PATH_ECHO_CHARS)}. Dictation sockets are created "
            "by the kitty fork as <session>/dictate-<pane>.sock.")
    return target


def _protocol_version(raw: object) -> str:
    """Return the caller's declared version, or raise on an incompatible major.

    Accepts "1", "1.0", "1.7" and the integer 1. Refuses a different major.
    """
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise ProtocolError(
            f"'v' must be a version string such as {PROTOCOL_VERSION!r}, got "
            f"{type(raw).__name__}. Omit it to accept whatever this daemon "
            "speaks.")
    text = str(raw)
    match = _VERSION_TOKEN.fullmatch(text)
    if match is None:
        raise ProtocolError(
            f"'v' is {_echo(text)}; expected MAJOR or MAJOR.MINOR, for example "
            f"{PROTOCOL_VERSION!r}.")
    major = int(match.group(1))
    if major != PROTOCOL_MAJOR:
        # Well-formed, and refused because this daemon does not speak it: that
        # is `unsupported`, not `malformed`. A token that is not a version at
        # all (above) stays malformed.
        raise ProtocolError(
            f"protocol major {major} is not supported; this daemon speaks "
            f"{PROTOCOL_SCHEMA} (major {PROTOCOL_MAJOR}, current "
            f"{PROTOCOL_VERSION}). A different major means a request shape "
            "changed meaning; upgrade the client rather than retrying.",
            code=ERR_UNSUPPORTED)
    return text


def _deadline_ms(raw: object) -> int:
    """Return a positive relative deadline in milliseconds, or raise."""
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ProtocolError(
            f"'deadline_ms' must be an integer number of milliseconds, got "
            f"{type(raw).__name__}. Send for example 5000 for five seconds.")
    if raw <= 0:
        raise ProtocolError(
            f"'deadline_ms' must be greater than zero, got {_echo(raw)}. A "
            "deadline that has already passed cannot be met; omit it to wait "
            "indefinitely.")
    if raw > MAX_DEADLINE_MS:
        raise ProtocolError(
            f"'deadline_ms' is {_echo(raw)}; the ceiling is {MAX_DEADLINE_MS} "
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
    # Two different refusals that used to share one message and, on the wire,
    # the code `internal`. A missing or non-string op is a request that is not
    # shaped like one: malformed. A string this daemon does not implement is a
    # well-formed request for something it does not support.
    if not isinstance(op, str):
        raise ProtocolError(
            f"'op' is required and must be a string, got "
            f"{type(op).__name__}. Use one of: {', '.join(OPS)}.")
    if op not in OPS:
        raise ProtocolError(
            f"unknown op {_echo(op)}. Use one of: {', '.join(OPS)}.",
            code=ERR_UNSUPPORTED)
    request: dict = {"op": op, "id": _request_id(msg.get("id"))}
    if "v" in msg:
        request["v"] = _protocol_version(msg.get("v"))
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
        if "chunk_sock" in msg:
            # A15: streamed synthesis needs somewhere to stream TO. It reuses
            # _validated_socket, so the path containment already reviewed for
            # dictation applies unchanged rather than a second, weaker set
            # being invented.
            #
            # Deliberately NOT called "sock". The pinned regression test
            # tests/test_protocol.py:374 asserts that a speak request DROPS an
            # unused "sock" field, and speak claiming that name would change a
            # shipped behaviour -- a contract change belonging to the successor
            # seam freeze. A new capability gets a new name; the old field goes
            # on being dropped exactly as before.
            request["chunk_sock"] = _validated_socket(
                msg.get("chunk_sock"), session_dir)
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
    return {"ok": False, "error": _cut_prose(message, MAX_ERROR_PROSE_CHARS),
            "code": code}


def check_audio_bytes(length: int, *, embedded: bool = False) -> int:
    """Return ``length`` if it is a permissible audio size, else refuse.

    A06 and A07. ``embedded`` applies the much smaller in-JSON ceiling: audio
    that large belongs behind a descriptor, not inside a control message.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ProtocolError(
            f"audio length must be a non-negative integer, got {length!r}.")
    limit = MAX_EMBEDDED_AUDIO_BYTES if embedded else MAX_AUDIO_BYTES
    if length > limit:
        raise MessageTooLarge(
            f"audio is {length} bytes; the "
            f"{'embedded-in-JSON' if embedded else 'audio'} limit is {limit}. "
            + ("Send a descriptor or a path instead of the samples."
               if embedded else
               "Split the clip or stream it; it is refused, not truncated."))
    return length


def _segment_id(raw: object) -> str:
    if not isinstance(raw, str) or not raw or len(raw) > MAX_ID_CHARS:
        raise ProtocolError(
            f"segment id must be 1-{MAX_ID_CHARS} characters, got {raw!r}.")
    return raw


def dictation_partial(text: str, segment: str | None = None) -> dict:
    """Return an in-progress recognition datagram.

    A09: an UNSTABLE segment id. The same id may be re-sent with revised text;
    a consumer keys on it to replace rather than append.
    """
    datagram: dict = {"partial": text}
    if segment is not None:
        datagram["segment"] = _segment_id(segment)
        datagram["stable"] = False
    return datagram


def dictation_final(text: str, segment: str | None = None,
                    words: list | None = None,
                    speaker: dict | None = None) -> dict:
    """Return the final recognition datagram for one turn.

    A09 stable segment id, A10 word timestamps, A11 speaker label with a
    confidence. Each is optional so an engine that cannot supply it omits the
    key rather than inventing a value -- a fabricated timestamp is worse than
    an absent one.
    """
    datagram: dict = {"final": text}
    if segment is not None:
        datagram["segment"] = _segment_id(segment)
        datagram["stable"] = True
    if words is not None:
        datagram["words"] = _words(words)
    if speaker is not None:
        datagram["speaker"] = _speaker(speaker)
    return datagram


def _words(raw: object) -> list:
    """A10: [{word, start_ms, end_ms}], monotonic and non-negative."""
    if not isinstance(raw, list):
        raise ProtocolError(
            f"'words' must be a list, got {type(raw).__name__}.")
    out = []
    previous_end = -1
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ProtocolError(f"word {index} must be an object.")
        word = item.get("word")
        start = item.get("start_ms")
        end = item.get("end_ms")
        if not isinstance(word, str) or not word:
            raise ProtocolError(f"word {index} needs a non-empty 'word'.")
        for name, value in (("start_ms", start), ("end_ms", end)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProtocolError(
                    f"word {index} '{name}' must be a non-negative integer of "
                    f"milliseconds, got {value!r}.")
        if end < start:
            raise ProtocolError(
                f"word {index} ends at {end}ms before it starts at {start}ms.")
        if start < previous_end:
            raise ProtocolError(
                f"word {index} starts at {start}ms, before word {index - 1} "
                f"ended at {previous_end}ms; timestamps must not go backwards.")
        previous_end = end
        out.append({"word": word, "start_ms": start, "end_ms": end})
    return out


def _speaker(raw: object) -> dict:
    """A11: {label, confidence} with confidence in [0.0, 1.0]."""
    if not isinstance(raw, dict):
        raise ProtocolError(
            f"'speaker' must be an object, got {type(raw).__name__}.")
    label = raw.get("label")
    confidence = raw.get("confidence")
    if not isinstance(label, str) or not label or len(label) > MAX_ID_CHARS:
        raise ProtocolError(
            f"speaker label must be 1-{MAX_ID_CHARS} characters, got {label!r}.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ProtocolError(
            "A11 requires a speaker confidence; a label without one cannot be "
            f"acted on. Got {confidence!r}.")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ProtocolError(
            f"speaker confidence must be within [0.0, 1.0], got {confidence}.")
    return {"label": label, "confidence": float(confidence)}


def dictation_error(message: str, code: str | None = None) -> dict:
    """Return a dictation failure datagram, with a closed code when given.

    P12 applies to this channel too, and it was left prose-only when
    reply_error was closed. The code is OPTIONAL rather than defaulted, for one
    reason: the pinned regression test at tests/test_protocol.py:419 asserts
    that an error datagram has exactly the key {"error"}. Defaulting a code in
    would change that shipped wire shape, which is a contract change belonging
    to the successor seam freeze, not to this commit. Runtime callers pass one;
    the bare form is unchanged.
    """
    message = _cut_prose(message, MAX_ERROR_PROSE_CHARS)
    if code is None:
        return {"error": message}
    if code not in ERROR_CODES:
        raise ProtocolError(
            f"unknown error code {code!r}. Use one of: "
            f"{', '.join(ERROR_CODES)}.")
    return {"error": message, "code": code}


def synthesis_chunk(sequence: int, *, pcm_bytes: int, sample_rate: int,
                    voice: str, model: str, seed: int | None = None,
                    final: bool = False, **settings: object) -> dict:
    """Return one streamed synthesis chunk descriptor.

    A12 sequence, A13 voice provenance, A14 seed and deterministic settings.
    The samples never ride in the JSON -- only their length, which
    check_audio_bytes bounds.

    WHAT THIS IS NOT, stated because the earlier docstring overclaimed it and an
    independent review caught that: this descriptor carries length, rate and
    provenance and NO path, handle or retrieval token for the PCM. The audio
    goes to the daemon's local player. So a subscriber learns that a bounded
    clip is playing and what produced it -- an ANNOUNCEMENT of local playback,
    arriving before the utterance completes. It is not delivery of playable
    audio to the subscriber, and A15's positive arm is not satisfied by it. A
    playable transport, or a narrower A15, is still owed.
    """
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ProtocolError(
            f"chunk 'sequence' must be a non-negative integer, got {sequence!r}.")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ProtocolError(
            f"chunk 'sample_rate' must be greater than zero, got {sample_rate!r}.")
    if not isinstance(voice, str) or not _VOICE_TOKEN.fullmatch(voice):
        raise ProtocolError(
            "chunk 'voice' must be 1-32 characters from [A-Za-z0-9_+-].")
    if not isinstance(model, str) or not model:
        raise ProtocolError("chunk 'model' must be a non-empty string.")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ProtocolError(f"chunk 'seed' must be an integer, got {seed!r}.")
    chunk: dict = {
        "sequence": sequence,
        "pcm_bytes": check_audio_bytes(pcm_bytes),
        "sample_rate": sample_rate,
        "voice": voice,
        "model": model,
        "final": bool(final),
    }
    if seed is not None:
        chunk["seed"] = seed
    if settings:
        chunk["settings"] = dict(settings)
    return chunk
