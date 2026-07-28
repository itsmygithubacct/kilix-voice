"""Speech synthesis, and the text conditioning that has to happen first.

Read-aloud is a pipeline: raw pane text → ``condition_text`` →
``SentenceChunker`` → ``EspeakTts.synth`` → the Player.  It is split there so
that audio starts on the first sentence rather than the last, and so a stop
request lands within one clip instead of at the end of a screenful.

The engines are synchronous primitives: ``synth`` runs one process, waits for
it, and returns PCM.  Nothing here opens an audio device — espeak-ng is asked
for a WAV on stdout, which is parsed in memory — so this module is safe to
import and to exercise on a machine with no sound server at all.
"""

from __future__ import annotations

import re
import subprocess

from . import settings, util

# espeak-ng writes 22.05 kHz mono at --stdout. The real rate always comes from
# the WAV header; this is only what an empty clip is labelled with.
ESPEAK_SAMPLE_RATE = 22050

# Synthesis is far faster than real time, so a run that takes this long is
# stuck rather than busy. Generous, because a first run pages in the voice data.
SYNTH_TIMEOUT_BASE_S = 20.0
SYNTH_TIMEOUT_PER_CHAR_S = 0.002

# The hard ceiling on one chunk, in characters: roughly ten seconds of speech.
# It exists so an unpunctuated wall of terminal output still starts playing
# promptly, and so the chunker cannot grow a buffer without bound.
MAX_CHUNK_CHARS = 240

TRUNCATION_NOTE = " …truncated"

INSTALL_HINT = ("Install it (Debian/Ubuntu: sudo apt install espeak-ng; "
                "Fedora: sudo dnf install espeak-ng), or set "
                f"{settings.KEY_TTS_ENGINE}=off to silence read-aloud")

# Mirrors the settings vocabulary for KILIX_VOICE_TTS_VOICE: a voice name
# becomes argv for espeak-ng, so it is held to an alphabet even when a caller
# passes it directly instead of through the shared settings file.
_VOICE_TOKEN = re.compile(r"^[A-Za-z0-9_+-]{1,32}$")


class TtsError(RuntimeError):
    """Synthesis failed; the message says what to do about it."""


# --------------------------------------------------------------------------
# Conditioning
# --------------------------------------------------------------------------

# The introducers that begin something to delete. ESC (\x1b) covers the 7-bit
# forms; the rest are the 8-bit C1 equivalents, which appear when terminal
# bytes were decoded as latin-1 somewhere upstream.
_INTRODUCER = re.compile(r"[\x1b\x90\x98\x9b\x9d\x9e\x9f]")

# ESC ] (OSC), ESC _ (APC — kitty graphics), ESC P (DCS), ESC ^ (PM),
# ESC X (SOS): an introducer whose payload runs to a string terminator.
_STRING_OPENERS = frozenset("]_P^X")
_C1_STRING_OPENERS = frozenset("\x90\x98\x9d\x9e\x9f")

# ST in both forms, plus BEL: BEL only terminates OSC by the standard, but
# accepting it everywhere costs nothing (no payload we care about, base64
# included, can contain it) and salvages malformed output.
_STRING_TERMINATOR = re.compile(r"[\x07\x9c]|\x1b\\")

# Runs of three or more of the *same* box-drawing character (U+2500-U+257F): a
# rule, a border, or a table edge. Two in a row can still be meaningful, so
# they survive.
_BOX_RUN = re.compile("([\u2500-\u257f])\\1{2,}")

_HORIZONTAL_RUN = re.compile(r"[^\S\n]+")
_NEWLINE_RUN = re.compile(r" ?\n[ \n]*")

# What is left of a line once the escapes are gone: C0 and C1 controls, DEL,
# and lone surrogates. Surrogates matter because a str decoded with
# errors="surrogateescape" carries them and cannot be encoded back to UTF-8
# for the engine's stdin — they would fail the read, not the character.
_CONTROL_TRANSLATION: dict[int, str | None] = {
    code: None for code in range(0x00, 0x20)}
_CONTROL_TRANSLATION.update({code: None for code in range(0x80, 0xa0)})
_CONTROL_TRANSLATION.update({code: None for code in range(0xd800, 0xe000)})
_CONTROL_TRANSLATION[0x7f] = None
_CONTROL_TRANSLATION[0x09] = " "
_CONTROL_TRANSLATION[0x0a] = "\n"
_CONTROL_TRANSLATION[0x0b] = "\n"
_CONTROL_TRANSLATION[0x0c] = "\n"
_CONTROL_TRANSLATION[0x0d] = "\n"


def _as_text(value: object) -> str:
    """Coerce whatever a pane capture hands us into a str, never raising."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace")
    try:
        return str(value)
    except Exception:  # a __str__ that raises is still not our failure to have
        return ""


def _skip_string(text: str, start: int) -> int:
    """Return the index just past a string-terminated payload.

    An unterminated one runs to the end of the text on purpose: a truncated
    capture of a kitty graphics APC leaves megabytes of base64 with no
    terminator, and reading that aloud is worse than losing the tail.
    """
    match = _STRING_TERMINATOR.search(text, start)
    return match.end() if match else len(text)


def _skip_csi(text: str, start: int) -> int:
    """Return the index just past a CSI sequence's parameters and final byte."""
    index = start
    limit = len(text)
    while index < limit and "\x20" <= text[index] <= "\x3f":
        index += 1
    if index < limit and "\x40" <= text[index] <= "\x7e":
        return index + 1
    # No final byte: the sequence was cut short by the capture or by a control
    # character. Drop what was scanned and let the rest be read normally.
    return index


def _skip_sequence(text: str, start: int) -> int:
    """Return the index just past the escape sequence beginning at ``start``."""
    opener = text[start]
    index = start + 1
    if opener == "\x9b":
        return _skip_csi(text, index)
    if opener in _C1_STRING_OPENERS:
        return _skip_string(text, index)
    if index >= len(text):
        return index                      # a trailing ESC introduces nothing
    following = text[index]
    index += 1
    if following == "[":
        return _skip_csi(text, index)
    if following in _STRING_OPENERS:
        return _skip_string(text, index)
    if "\x20" <= following <= "\x2f":
        # ESC ( B, ESC # 8 and friends: intermediates then one final byte.
        while index < len(text) and "\x20" <= text[index] <= "\x2f":
            index += 1
        return min(index + 1, len(text))
    if "\x30" <= following <= "\x7e":
        return index                      # two-character escape: ESC 7, ESC c
    return start + 1                      # a stray ESC before a control byte


def _strip_escapes(text: str) -> str:
    """Remove every escape sequence, with its payload, from ``text``."""
    parts: list[str] = []
    pos = 0
    while True:
        match = _INTRODUCER.search(text, pos)
        if match is None:
            parts.append(text[pos:])
            return "".join(parts)
        parts.append(text[pos:match.start()])
        pos = _skip_sequence(text, match.start())


def _budget(max_chars: int | None) -> int | None:
    """Return a usable character budget, or None for unlimited."""
    if max_chars is None:
        return None
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        return None
    return max(limit, 0)


def condition_text(text: str, *, max_chars: int | None) -> str:
    """Return pane text reduced to something worth reading aloud.

    The steps run in this order, and the order is load-bearing:

    1. escape sequences and their payloads go first, so a CSI parameter byte or
       a megabyte of kitty graphics base64 is never mistaken for text later;
    2. the control characters and lone surrogates left over go with them;
    3. box-drawing rules are removed *before* whitespace is collapsed, so the
       gap a removed border leaves behind closes up instead of being spoken as
       a pause;
    4. the character budget is applied last, to the text that will actually be
       spoken rather than to the decoration that was thrown away.

    It never raises: the input is whatever a pane happened to be showing.
    """
    result = _strip_escapes(_as_text(text))
    result = result.translate(_CONTROL_TRANSLATION)
    result = _BOX_RUN.sub("", result)
    # Every horizontal run becomes exactly one space first, which is what lets
    # the second pattern describe line breaks with a single optional space.
    result = _HORIZONTAL_RUN.sub(" ", result)
    result = _NEWLINE_RUN.sub("\n", result)
    result = result.strip()

    limit = _budget(max_chars)
    if limit is None or len(result) <= limit:
        return result
    if limit == 0:
        return ""
    return result[:limit].rstrip() + TRUNCATION_NOTE


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

_TERMINATORS = frozenset(".!?…")
_CLOSERS = frozenset(")]}\"'”’»")

# Deliberately short. Every entry here *suppresses* a split, so a word that is
# often the last one in a sentence ("etc.", "al.", "Inc.") must not be listed:
# a missed sentence end delays audio, a spurious one only shortens a clip.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "mx", "dr", "prof", "sr", "jr", "st", "mt",
    "eg", "ie", "cf", "vs", "fig", "dept", "approx", "vol",
})


def _is_abbreviation(before: str) -> bool:
    """Return whether the word ending at a '.' is one to keep reading past."""
    index = len(before)
    while index > 0 and (before[index - 1].isalpha()
                         or before[index - 1] == "."):
        index -= 1
    token = before[index:].replace(".", "").lower()
    # A single letter is an initial ("J. R. R. Tolkien"), not a sentence end.
    return len(token) == 1 or token in _ABBREVIATIONS


def _force_cut(buffer: str, start: int, limit: int) -> int:
    """Return where to break a chunk that has run past MAX_CHUNK_CHARS."""
    space = buffer.rfind(" ", start, limit)
    return space + 1 if space > start else limit


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Return (complete sentences, the text still waiting for its end).

    One left-to-right pass: each cut only ever moves ``start`` forward, so
    feeding a large capture costs the same as feeding it a line at a time.
    """
    pieces: list[str] = []
    start = 0
    index = 0
    limit = len(buffer)
    while index < limit:
        char = buffer[index]
        if index - start >= MAX_CHUNK_CHARS:
            cut = _force_cut(buffer, start, index)
        elif char == "\n":
            # A line of terminal output is a unit whether or not it is a
            # sentence; `total 92` never ends in a full stop.
            cut = index + 1
        elif char in _TERMINATORS:
            after = index + 1
            while after < limit and buffer[after] in _CLOSERS:
                after += 1
            if after >= limit:
                # Whether this ends a sentence depends on the character after
                # it, which has not arrived yet. flush() resolves it.
                break
            if not buffer[after].isspace() or (
                    char == "." and _is_abbreviation(buffer[start:index])):
                index += 1
                continue
            cut = after
        else:
            index += 1
            continue
        piece = buffer[start:cut].strip()
        if piece:
            pieces.append(piece)
        start = cut
        index = cut
    return pieces, buffer[start:]


class SentenceChunker:
    """Splits conditioned text into clips that can be synthesised as they land.

    ``feed`` returns only what is certainly complete, so the daemon can start
    playing the first sentence while the rest of the screen is still being
    conditioned, and a stop request never has to wait for more than one clip.
    ``flush`` yields whatever is left when the text ends.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add ``text`` and return every complete sentence now available."""
        self._buffer += _as_text(text)
        pieces, self._buffer = _split_sentences(self._buffer)
        return pieces

    def flush(self) -> str:
        """Return the unterminated tail and reset; "" when none is pending."""
        tail = self._buffer.strip()
        self._buffer = ""
        return tail


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------


class NullTts:
    """The engine for `KILIX_VOICE_TTS_ENGINE=off`: silence, not failure.

    Read-aloud that is switched off must still answer a speak request, so the
    daemon's dispatch does not need a special case and the TUIs can show a
    working pipeline with no audio at the end of it.
    """

    name = "null"

    def synth(self, text: str) -> tuple[bytes, int]:
        """Return an empty clip regardless of ``text``."""
        return b"", ESPEAK_SAMPLE_RATE


def espeak_binary() -> str | None:
    """Return the synthesiser on PATH, or None when none is installed.

    espeak-ng is preferred; the older espeak accepts the same options and is
    what some distributions still ship.
    """
    return util.which("espeak-ng") or util.which("espeak")


def build_synth_cmd(cfg: dict | None, *, voice: str, rate: int) -> list[str]:
    """Return the argv that synthesises one clip as a WAV on stdout.

    ``cfg`` may replace the whole command with ``tts.cmd`` (a list of strings,
    with "{voice}" and "{rate}" substituted). That override is the only way a
    process other than espeak-ng can be reached from this module, which makes
    it the seam a test injects a fake engine through.
    """
    override = util.cfg_get(cfg or {}, "tts.cmd")
    if override:
        if (not isinstance(override, (list, tuple)) or not override
                or not all(isinstance(part, str) for part in override)):
            raise TtsError(
                f"tts.cmd must be a non-empty list of strings, got "
                f"{override!r}. Use for example: "
                '["espeak-ng", "-v", "{voice}", "-s", "{rate}", "--stdout"].')
        return [part.replace("{voice}", voice).replace("{rate}", str(rate))
                for part in override]
    binary = espeak_binary()
    if binary is None:
        raise TtsError(
            "no speech synthesiser found: neither espeak-ng nor espeak is on "
            f"PATH. {INSTALL_HINT}.")
    # -b 1 forces the UTF-8 reading of stdin rather than letting espeak guess
    # from the first bytes of a chunk that may not carry a hint.
    return [binary, "-b", "1", "-v", voice, "-s", str(rate), "--stdout"]


def _stderr_note(err: bytes) -> str:
    """Return the engine's own complaint, trimmed for a one-line message."""
    text = err.decode("utf-8", "replace").strip()
    if not text:
        return ""
    return f" ({text.splitlines()[0].strip()[:200]})"


def _failure_hint(binary: str, voice: str) -> str:
    """Return what to do about a synthesis run that failed with ``voice``."""
    if voice.startswith("mb-"):
        return (f"Install the mbrola voice for {voice!r} (Debian/Ubuntu: sudo "
                "apt install mbrola mbrola-us1), or set "
                f"{settings.KEY_TTS_ENGINE}=espeak")
    return f"Check that the voice {voice!r} exists: {binary} --voices"


class EspeakTts:
    """espeak-ng synthesis: one process per clip, WAV parsed in memory.

    Reading the WAV from stdout rather than writing a file keeps synthesis free
    of temporary files, and means the sample rate is whatever the engine chose
    for this clip — the caller is told it rather than assuming it.
    """

    name = "espeak"

    def __init__(self, cfg: dict | None = None, *, voice: str | None = None,
                 rate: int | None = None, mbrola: bool = False) -> None:
        self._cfg = cfg or {}
        self.voice = self._checked_voice(
            settings.tts_voice() if voice is None else voice)
        self.rate = int(settings.tts_rate() if rate is None else rate)
        self.mbrola = bool(mbrola)
        # Set when an mbrola voice turns out not to be installed, so a TUI can
        # explain why the voice sounds like plain espeak.
        self.mbrola_error = ""
        self._mbrola_ok = self.mbrola

    @staticmethod
    def _checked_voice(voice: str) -> str:
        token = str(voice).strip()
        if not _VOICE_TOKEN.match(token):
            raise TtsError(
                f"invalid voice name {voice!r}: expected 1-32 characters from "
                "[A-Za-z0-9_+-]. Use a name the engine lists, such as en-us "
                "(run: espeak-ng --voices).")
        return token

    def synth(self, text: str) -> tuple[bytes, int]:
        """Return (s16le mono PCM, sample rate) for one clip of ``text``."""
        clean = _as_text(text).strip()
        if not clean:
            # A screen of nothing but decoration conditions down to nothing;
            # that is an empty clip, not a failure to synthesise.
            return b"", ESPEAK_SAMPLE_RATE
        if self._mbrola_ok:
            try:
                return self._run(clean, f"mb-{self.voice}")
            except TtsError as error:
                # A quality tier that is not installed must never lose a read.
                # Remembering the failure keeps the rest of the page from
                # paying for a doomed process once per sentence.
                self._mbrola_ok = False
                self.mbrola_error = str(error)
        return self._run(clean, self.voice)

    def _run(self, text: str, voice: str) -> tuple[bytes, int]:
        command = build_synth_cmd(self._cfg, voice=voice, rate=self.rate)
        timeout = SYNTH_TIMEOUT_BASE_S + len(text) * SYNTH_TIMEOUT_PER_CHAR_S
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        except OSError as error:
            raise TtsError(
                f"cannot run {command[0]!r}: {error}. Check that it exists and "
                f"is executable; if this is the stock engine, (re)install "
                f"espeak-ng.") from error
        try:
            out, err = process.communicate(
                text.encode("utf-8", "replace"), timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise TtsError(
                f"{command[0]} did not finish within {timeout:.1f}s for "
                f"{len(text)} characters of text. Read a smaller extent by "
                f"lowering {settings.KEY_TTS_MAX_CHARS}, or check whether the "
                f"engine is blocked on an audio device.") from error

        if process.returncode != 0:
            raise TtsError(
                f"{command[0]} exited {process.returncode}"
                f"{_stderr_note(err)}. "
                f"{_failure_hint(command[0], voice)}.")
        try:
            # Trusting the header rather than a fixed rate: mbrola voices and
            # espeak builds do not all synthesise at the same rate.
            return util.parse_wav_bytes(out)
        except ValueError as error:
            raise TtsError(
                f"{command[0]} produced no usable audio: {error}"
                f"{_stderr_note(err)} "
                f"{_failure_hint(command[0], voice)}.") from error


def make_tts(cfg: dict | None = None) -> NullTts | EspeakTts:
    """Return the engine the shared settings select.

    Construction deliberately does not probe for espeak-ng: a missing
    synthesiser has to degrade the read at the moment it is asked for, with a
    message saying how to install it, rather than stop a TUI or the daemon
    from starting at all.
    """
    engine = settings.tts_engine()
    if engine == "off":
        return NullTts()
    # settings.tts_engine() validates against the vocabulary, so anything that
    # is not "off" is espeak, with or without the mbrola tier on top.
    return EspeakTts(cfg, mbrola=(engine == "mbrola"))
