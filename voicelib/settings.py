"""Reader and writer for the shared GPU Terminal settings document.

kilix-voice has no configuration file of its own.  Every knob lives in the same
``KEY=value`` document Kilix's SDK writes (``settings.conf``), so the terminal,
the desktop, and these TUIs always read one source of truth.  The file is
deliberately not shell code: it is parsed, never executed.

Because the file is shared, writes are conservative — an update rewrites only
the keys it was given and preserves every comment and foreign key verbatim.
Reads are forgiving in the other direction: an unrecognised value falls back to
this module's default rather than being coerced into something adjacent, so a
newer Kilix writing a value we do not know cannot make us guess.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping

from . import paths


class SettingsError(RuntimeError):
    """An unusable settings file, an unknown key, or an invalid write."""


KEY_SPEAK = "KILIX_CHROME_SPEAK"
KEY_DICTATE = "KILIX_CHROME_DICTATE"
KEY_TTS_ENGINE = "KILIX_VOICE_TTS_ENGINE"
KEY_TTS_VOICE = "KILIX_VOICE_TTS_VOICE"
KEY_TTS_RATE = "KILIX_VOICE_TTS_RATE"
KEY_TTS_EXTENT = "KILIX_VOICE_TTS_EXTENT"
KEY_TTS_MAX_CHARS = "KILIX_VOICE_TTS_MAX_CHARS"
KEY_STT_ENGINE = "KILIX_VOICE_STT_ENGINE"
KEY_STT_MODEL = "KILIX_VOICE_STT_MODEL"
KEY_STT_SUBMIT = "KILIX_VOICE_STT_SUBMIT"
KEY_STT_MAX_SECONDS = "KILIX_VOICE_STT_MAX_SECONDS"
KEY_STT_SILENCE_MS = "KILIX_VOICE_STT_SILENCE_MS"
KEY_STT_PUNCTUATION = "KILIX_VOICE_STT_PUNCTUATION"
KEY_DEVICE_IN = "KILIX_VOICE_DEVICE_IN"
KEY_DEVICE_OUT = "KILIX_VOICE_DEVICE_OUT"
KEY_HISTORY = "KILIX_VOICE_HISTORY"

# key -> (default, choices or None). Numeric settings are presets rather than
# free-form numbers so a dropdown, a TUI cycle and the file itself all share one
# vocabulary, and an out-of-vocabulary value reads back as the default.
SPEC: dict[str, tuple[str, tuple[str, ...] | None]] = {
    KEY_SPEAK: ("1", None),
    KEY_DICTATE: ("1", None),
    KEY_TTS_ENGINE: ("espeak", ("espeak", "mbrola", "off")),
    KEY_TTS_VOICE: ("en-us", None),
    KEY_TTS_RATE: ("170", ("120", "150", "170", "200", "240")),
    KEY_TTS_EXTENT: ("screen", ("screen", "scrollback", "selection")),
    KEY_TTS_MAX_CHARS: ("4000", ("1000", "4000", "16000", "unlimited")),
    KEY_STT_ENGINE: ("vosk", ("vosk", "vibevoice", "off")),
    KEY_STT_MODEL: ("small-en-us",
                    ("small-en-us", "lgraph-en-us", "vibevoice-asr-bitnet")),
    # There is deliberately no "always": dictation never presses Enter for you.
    KEY_STT_SUBMIT: ("never", ("never", "confirm")),
    KEY_STT_MAX_SECONDS: ("30", ("15", "30", "60", "120")),
    KEY_STT_SILENCE_MS: ("900", ("500", "900", "1500")),
    KEY_STT_PUNCTUATION: ("1", None),
    KEY_DEVICE_IN: ("default", None),
    KEY_DEVICE_OUT: ("default", None),
    KEY_HISTORY: ("off", ("off", "on")),
}

BOOL_KEYS = frozenset({KEY_SPEAK, KEY_DICTATE, KEY_STT_PUNCTUATION})

# Values without a fixed vocabulary still become argv for espeak-ng, parec and
# pacat, so they are held to an alphabet instead of being passed through raw.
_VOICE_PATTERN = re.compile(r"^[A-Za-z0-9_+-]{1,32}$")
_DEVICE_PATTERN = re.compile(r"^[A-Za-z0-9_.:+-]{1,128}$")
_PATTERNS = {
    KEY_TTS_VOICE: _VOICE_PATTERN,
    KEY_DEVICE_IN: _DEVICE_PATTERN,
    KEY_DEVICE_OUT: _DEVICE_PATTERN,
}

_FALSE_WORDS = ("", "0", "no", "false", "off", "disabled")
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# Kilix's own writer stamps this header on a file it creates; matching it keeps
# a file first written by us recognisable to the rest of the stack.
SETTINGS_HEADER = "# GPU Terminal shared settings (KEY=value; not shell code)."
SETTINGS_MARKER = "# -- Kilix voice --"


def truthy(value_: object) -> bool:
    """Return the shared stack's boolean reading of a settings value."""
    return str(value_).strip().lower() not in _FALSE_WORDS


def defaults() -> dict[str, str]:
    """Return the default value of every key this repo understands."""
    return {key: default for key, (default, _choices) in SPEC.items()}


def parse_text(text: str) -> dict[str, str]:
    """Parse a settings document; the last assignment of a key wins."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _ASSIGNMENT.match(line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def read_text(path: str | None = None) -> tuple[str, bool]:
    """Return (contents, existed). A missing or unreadable file is not fatal."""
    target = path or paths.settings_file()
    try:
        with open(target, encoding="utf-8", errors="replace") as stream:
            return stream.read(), True
    except OSError:
        return "", False


def load(path: str | None = None) -> dict[str, str]:
    """Return the raw effective values: file contents over our defaults.

    Foreign keys present in the file are kept, so a caller can read a Kilix
    setting from the same snapshot without a second parse.  Values are returned
    exactly as written — validation belongs to ``value()``.
    """
    text, exists = read_text(path)
    values = defaults()
    if exists:
        values.update(parse_text(text))
    return values


def _spec(key: str) -> tuple[str, tuple[str, ...] | None]:
    try:
        return SPEC[key]
    except KeyError as error:
        raise SettingsError(
            f"unknown kilix-voice setting {key!r}. Valid keys are: "
            f"{', '.join(sorted(SPEC))}.") from error


def value(key: str, path: str | None = None) -> str:
    """Return the validated value of ``key``, or its default."""
    default, choices = _spec(key)
    raw = str(load(path).get(key, default)).strip()
    if choices is not None:
        # Case-insensitive matching only; an out-of-vocabulary value is never
        # nudged towards the nearest choice.
        lowered = raw.lower()
        return lowered if lowered in choices else default
    pattern = _PATTERNS.get(key)
    if pattern is not None and not pattern.match(raw):
        return default
    return raw


def enabled(key: str, path: str | None = None) -> bool:
    """Return the boolean reading of a validated value."""
    return truthy(value(key, path))


def tts_engine(path: str | None = None) -> str:
    return value(KEY_TTS_ENGINE, path)


def tts_voice(path: str | None = None) -> str:
    return value(KEY_TTS_VOICE, path)


def tts_rate(path: str | None = None) -> int:
    """Return the read-aloud speed in words per minute."""
    return int(value(KEY_TTS_RATE, path))


def tts_extent(path: str | None = None) -> str:
    """Return what "read this" means: screen, scrollback, or selection."""
    return value(KEY_TTS_EXTENT, path)


def tts_max_chars(path: str | None = None) -> int | None:
    """Return the read-aloud character budget; None means unlimited."""
    token = value(KEY_TTS_MAX_CHARS, path)
    return None if token == "unlimited" else int(token)


def stt_engine(path: str | None = None) -> str:
    return value(KEY_STT_ENGINE, path)


def stt_model(path: str | None = None) -> str:
    return value(KEY_STT_MODEL, path)


def stt_submit(path: str | None = None) -> str:
    """Return the submission policy: "never" or "confirm", never "always"."""
    return value(KEY_STT_SUBMIT, path)


def stt_max_seconds(path: str | None = None) -> int:
    """Return the hard ceiling on one dictation turn."""
    return int(value(KEY_STT_MAX_SECONDS, path))


def stt_silence_ms(path: str | None = None) -> int:
    """Return the trailing silence that closes a dictation turn."""
    return int(value(KEY_STT_SILENCE_MS, path))


def device_in(path: str | None = None) -> str:
    return value(KEY_DEVICE_IN, path)


def device_out(path: str | None = None) -> str:
    return value(KEY_DEVICE_OUT, path)


def history_enabled(path: str | None = None) -> bool:
    """Return whether completed turns may be recorded on disk."""
    return enabled(KEY_HISTORY, path)


def _rendered(key: str, raw: object) -> str:
    """Return the on-disk form of a value, rejecting anything invalid.

    Writes are strict where reads are lenient: a settings screen that stores a
    typo would make every later read silently fall back to the default.
    """
    default, choices = _spec(key)
    text = str(raw).strip()
    if key in BOOL_KEYS:
        return "1" if truthy(raw) else "0"
    if choices is not None:
        lowered = text.lower()
        if lowered not in choices:
            raise SettingsError(
                f"{key} must be one of: {', '.join(choices)} — got {text!r}.")
        return lowered
    pattern = _PATTERNS.get(key)
    if pattern is not None and not pattern.match(text):
        raise SettingsError(
            f"{key} must match {pattern.pattern} — got {text!r}. Use "
            f"{default!r} for the system default.")
    return text


def _set_value(text: str, key: str, rendered: str) -> str:
    """Return ``text`` with ``key`` assigned, editing in place where present."""
    line = f"{key}={rendered}"
    # [ \t] rather than \s: \s matches newlines, so the match would start at an
    # earlier line and rewriting it would eat the blank lines above the key.
    pattern = re.compile(rf"^[ \t]*{re.escape(key)}=.*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if matches:
        # The last assignment is the effective one, so that is the one to edit.
        last = matches[-1]
        return text[:last.start()] + line + text[last.end():]
    if SETTINGS_MARKER not in text:
        text = text.rstrip("\n") + f"\n\n{SETTINGS_MARKER}\n"
    return text.rstrip("\n") + "\n" + line + "\n"


def _initial_text() -> str:
    """Return the minimal document to create when the shared file is absent.

    Only our own section is written: the rest of the stack's keys belong to the
    hosts that own them, and inventing their defaults here would freeze a copy
    that drifts.
    """
    return f"{SETTINGS_HEADER}\n\n{SETTINGS_MARKER}\n"


def _atomic_write(target: str, text: str) -> None:
    directory = os.path.dirname(target) or "."
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(target)}.", dir=directory)
    except OSError as error:
        raise SettingsError(
            f"cannot write the shared settings file in {directory}: {error}. "
            "Check that the directory exists and is writable, or point "
            "GPU_TERMINAL_SETTINGS_FILE at one that is.") from error
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            handle = -1
            stream.write(text)
            stream.flush()
            # A settings file half-written across a crash is worse than a stale
            # one, so the replacement is durable before it becomes visible.
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = ""
    except OSError as error:
        raise SettingsError(
            f"cannot write {target}: {error}. Check free space and the "
            "permissions on the containing directory.") from error
    finally:
        if handle >= 0:
            os.close(handle)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def update(changes: Mapping[str, object], path: str | None = None) -> str:
    """Atomically apply ``changes`` to the shared file; return its path.

    Comments, layout and every key this repo does not own survive untouched:
    the document belongs to Kilix, and a voice settings screen is only one of
    its writers.
    """
    target = path or paths.settings_file()
    rendered = {key: _rendered(key, raw) for key, raw in changes.items()}
    if os.path.islink(target):
        # Reading through a link is harmless, but replacing one would silently
        # move the stack's source of truth to wherever it points.
        raise SettingsError(
            f"refusing to write settings through the symlink {target}. Replace "
            "it with a real file, or set GPU_TERMINAL_SETTINGS_FILE to the "
            "path you actually want written.")
    text, exists = read_text(target)
    if not exists:
        text = _initial_text()
    for key, value_ in rendered.items():
        text = _set_value(text, key, value_)
    _atomic_write(target, text)
    return target
