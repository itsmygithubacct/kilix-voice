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

import json
import os
import re
import subprocess
import threading
import time
from typing import TYPE_CHECKING, NamedTuple

from . import models, paths, protocol, settings, util

if TYPE_CHECKING:
    from .qwen_provider import QwenProviderTts

# espeak-ng writes 22.05 kHz mono at --stdout. The real rate always comes from
# the WAV header; this is only what an empty clip is labelled with.
ESPEAK_SAMPLE_RATE = 22050
PIPER_SAMPLE_RATE = 22050
PIPER_VOICE = "en_US-kristin-medium"
PIPER_STATUS_TIMEOUT_S = 5.0
PIPER_SYNTH_TIMEOUT_S = 210.0
PIPER_ENV_COMMAND = "KILIX_PIPER_TTS"

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


def _bounded(cap: float, budget: float | None) -> float:
    """The timeout for a blocking engine call, under the caller's budget.

    R3 F01: every engine applied its OWN ceiling -- about 20 s plus text cost
    for eSpeak, 210 s for Piper, 5 s for Piper's status probe -- no matter how
    little the caller had left.  Nothing can poll a deadline while
    ``communicate`` is blocked, so an engine ceiling longer than the request
    budget is a hole the daemon cannot close from outside.  Whichever is
    smaller wins.

    ``budget`` is seconds remaining, or None for an unbounded caller.
    """
    if budget is None:
        return cap
    if budget <= 0:
        raise TtsDeadlineExceeded(
            "the request deadline elapsed before synthesis could start")
    return min(cap, budget)


def _budget_cut(cap: float, budget: float | None) -> bool:
    """True when the caller's budget, not the engine's own ceiling, is in force.

    R6 finding 1: a Piper probe failure was attributed to the deadline whenever
    the budget was shorter than the probe's ceiling, whether or not the budget
    was what ended it -- so "not installed" became "the deadline elapsed". The
    only place that knows WHICH bound cut a blocking call is the site that set
    it, so a TimeoutExpired is attributed there, with this, and nowhere else.
    Re-reading the clock afterwards is not a substitute: a genuine failure that
    returns just before the deadline and is inspected just after it reads as a
    deadline that did not cut anything.
    """
    return budget is not None and budget <= cap


class TtsError(RuntimeError):
    """Synthesis failed; the message says what to do about it.

    A missing synthesiser or provider is the common case, so the family is
    `unavailable` on the wire unless a subclass knows better.
    """

    code = protocol.ERR_UNAVAILABLE


class TtsUnsupported(TtsError):
    """The request asked an engine for something it cannot do at all."""

    code = protocol.ERR_UNSUPPORTED


class TtsDeadlineExceeded(TtsError):
    """The caller's budget -- and nothing else -- ended a probe or a synthesis.

    Raised only where that is known for certain: the budget was already spent
    before any process started, or a process was killed by a timeout whose
    value was the caller's budget rather than the engine's own ceiling. Every
    other failure stays a plain TtsError with the provider's own diagnosis.
    A subclass, so every existing `except TtsError` still catches it.
    """

    code = protocol.ERR_DEADLINE


# --------------------------------------------------------------------------
# MBROLA voices
# --------------------------------------------------------------------------

# The MBROLA voices espeak-ng knows, by the language each speaks, most preferred
# first. Generated from espeak-ng 1.52's voices/mb data: its `language <tag>
# <priority>` lines, lower priority first, then the voice id. A voice id names
# its diphone database by its first part, so de4-en speaks English with de4.
MBROLA_VOICES_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    'af': ('af1',),
    'ar': ('ar1', 'ar2'),
    'cs': ('cz1', 'cz2'),
    'de': ('de1', 'de2', 'de3', 'de4', 'de6', 'de5', 'de7', 'de8'),
    'el': ('gr2', 'gr1'),
    'en': ('en1', 'us2', 'us1', 'us3', 'de1-en', 'de2-en', 'de3-en', 'de4-en',
           'de5-en', 'de6-en', 'gr2-en', 'ro1-en', 'fr1-en', 'fr4-en', 'hu1-en',
           'nl2-en', 'sw2-en', 'af1-en', 'pl1-en', 'sw1-en'),
    'en-gb': ('en1',),
    'en-uk': ('en1',),
    'en-us': ('us1', 'us2', 'us3'),
    'es': ('es3', 'es4', 'es1', 'es2', 'mx1', 'mx2', 'vz1'),
    'es-es': ('es3', 'es4', 'es1', 'es2'),
    'es-mx': ('mx1', 'mx2'),
    'es-vz': ('vz1',),
    'et': ('ee1',),
    'fa': ('ir1',),
    'fr': ('fr1', 'fr4', 'fr2', 'fr3', 'fr6', 'fr7', 'fr5', 'ca1', 'ca2'),
    'fr-be': ('fr5',),
    'fr-ca': ('ca1', 'ca2'),
    'fr-fr': ('fr1', 'fr4', 'fr2', 'fr3', 'fr6'),
    'grc': ('de6-grc',),
    'he': ('hb1', 'hb2'),
    'hi': ('in1', 'in2'),
    'hr': ('cr1',),
    'hu': ('hu1',),
    'id': ('id1',),
    'is': ('ic1',),
    'it': ('it3', 'it4', 'it1', 'it2'),
    'ja': ('jp1', 'jp2', 'jp3'),
    'la': ('la1',),
    'lt': ('lt1', 'lt2'),
    'mi': ('nz1',),
    'ms': ('ma1',),
    'nl': ('nl2', 'nl1', 'nl3'),
    'pl': ('pl1',),
    'pt': ('pt1', 'br1', 'br2', 'br3', 'br4'),
    'pt-br': ('br1', 'br2', 'br3', 'br4'),
    'pt-pt': ('pt1',),
    'ro': ('ro1',),
    'sv': ('sw1', 'sw2'),
    'te': ('tl1',),
    'tr': ('tr1', 'tr2'),
    'xex': ('br1-xex', 'br4-xex', 'br2-xex', 'br3-xex'),
    'zh': ('cn1',),
}

# An explicit MBROLA voice id such as us1 or de4-en, as opposed to a language.
_MBROLA_VOICE_ID = re.compile(r"^[a-z]{2}[0-9]{1,2}(-[a-z]{2,3})?$")
# espeak-ng's own default when XDG_DATA_DIRS is unset.
_MBROLA_DEFAULT_DATA_DIRS = "/usr/local/share:/usr/share"


def mbrola_database_installed(database: str) -> bool:
    """True when espeak-ng would find the MBROLA diphone database ``database``.

    The search espeak-ng's mbrola wrapper makes, and nothing is launched: a
    regular file at <dir>/mbrola/<db>, <dir>/mbrola/<db>/<db> or
    <dir>/mbrola/voices/<db>, for each <dir> in XDG_DATA_DIRS.
    """
    raw = os.environ.get("XDG_DATA_DIRS") or _MBROLA_DEFAULT_DATA_DIRS
    for root in (part for part in raw.split(":") if part):
        base = os.path.join(root, "mbrola")
        for path in (os.path.join(base, database),
                     os.path.join(base, database, database),
                     os.path.join(base, "voices", database)):
            if os.path.isfile(path):
                return True
    return False


def installed_mbrola_voices() -> tuple[str, ...]:
    """Every MBROLA voice espeak-ng knows whose database is installed, sorted."""
    known = sorted({voice for voices in MBROLA_VOICES_BY_LANGUAGE.values()
                    for voice in voices})
    return tuple(voice for voice in known
                 if mbrola_database_installed(voice.split("-")[0]))


def resolve_mbrola_voice(voice: str) -> str:
    """Return the espeak-ng voice name, mb-<id>, that speaks ``voice`` with MBROLA.

    ``voice`` is an MBROLA voice id (us1, de4-en), used as named, or a language
    tag such as en-us, the shared default. A language resolves to the most
    preferred of its voices whose database is installed. The tier used to run
    mb-<voice> verbatim, and espeak-ng has no voice called mb-en-us, so
    model=mbrola failed as shipped even where a US English MBROLA voice was
    installed.

    Raises TtsUnsupported when no MBROLA voice speaks the language, and
    TtsError (unavailable) when none of the voices that do is installed. Both
    name the MBROLA voices that are installed.
    """
    token = str(voice).strip().lower()
    if _MBROLA_VOICE_ID.match(token):
        return f"mb-{token}"
    candidates = MBROLA_VOICES_BY_LANGUAGE.get(token, ())
    for candidate in candidates:
        if mbrola_database_installed(candidate.split("-")[0]):
            return f"mb-{candidate}"
    installed = ", ".join(installed_mbrola_voices()) or "none"
    if not candidates:
        raise TtsUnsupported(
            f"no MBROLA voice speaks {voice!r}. Name an installed MBROLA voice "
            f"({installed}) or a language one of them speaks, such as en-us.")
    packages = " ".join(f"mbrola-{database}" for database in list(dict.fromkeys(
        candidate.split("-")[0] for candidate in candidates))[:3])
    raise TtsError(
        f"no MBROLA voice for {voice!r} is installed: it is spoken by "
        f"{', '.join(candidates)}, and the installed MBROLA voices are: "
        f"{installed}. Install one (Debian/Ubuntu: sudo apt install mbrola "
        f"{packages}), or choose an installed voice.")


def mbrola_selection_detail(voice: str) -> str:
    """One status line: the MBROLA voice ``voice`` runs, or why plain espeak does."""
    try:
        return (f"MBROLA voice {resolve_mbrola_voice(voice)} speaks {voice}, with "
                "plain espeak if it fails")
    except TtsError as error:
        return f"plain espeak speaks {voice}: {error}"


class RenderedSpeech(NamedTuple):
    """One complete in-memory rendering, ready for a file container."""

    pcm: bytes
    sample_rate: int
    chunks: int
    model: str
    voice: str
    rate: int


class SynthesisProvenance(NamedTuple):
    """What actually produced one clip: A13 family and voice, A14 seed.

    Each real engine records one at the end of a SUCCESSFUL synth, so a failed
    attempt -- the mbrola voice that is not installed -- never leaves
    provenance behind, and the fallback's success overwrites it. The daemon
    reads it on the worker thread straight after synth and carries it with
    the clip, so a descriptor describes the clip it announces, not whatever
    the engine did last.
    """

    model: str
    voice: str
    # The seed that produced the clip, or None when there is none to report.
    # kilix-voice never makes one up: a descriptor carries a seed only when the
    # engine reports one, or when its output is reproducible without one.
    seed: int | None = None
    seed_consumed: bool = False
    reproducible: bool = False
    rate_wpm: int = 0


def clip_provenance(engine: object) -> SynthesisProvenance:
    """Return the provenance of ``engine``'s last clip.

    A real engine records it. One that does not -- a third-party engine or a
    test double -- is described from its attributes, each type-checked, with
    its seed marked not consumed and its output not reproducible, which claims
    nothing the engine did not say. An engine that reports no integer seed gets
    none: inventing 0 would present a seed that never produced the clip. A
    voice string is passed through as given:
    an unusable voice is refused where the descriptor is built, not papered
    over here.
    """
    recorded = getattr(engine, "last_provenance", None)
    if isinstance(recorded, SynthesisProvenance):
        return recorded
    effective = getattr(engine, "effective_model", None)
    model = getattr(engine, "model", None)
    if isinstance(effective, str) and effective:
        family = effective
    elif isinstance(model, str) and model:
        family = model
    else:
        family = "unset"
    voice = getattr(engine, "voice", None)
    seed = getattr(engine, "seed", None)
    rate = getattr(engine, "rate", None)
    return SynthesisProvenance(
        model=family,
        voice=voice if isinstance(voice, str) and voice else "unset",
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        seed_consumed=False,
        reproducible=False,
        rate_wpm=rate if isinstance(rate, int) and not isinstance(rate, bool) else 0)


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


def speech_chunks(text: str, *, max_chars: int | None) -> list[str]:
    """Condition one speech turn and return its bounded synthesis clips."""
    conditioned = condition_text(text, max_chars=max_chars)
    chunker = SentenceChunker()
    chunks = chunker.feed(conditioned)
    tail = chunker.flush()
    if tail:
        chunks.append(tail)
    return chunks


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
    model = "off"
    voice = ""
    rate = 0
    last_provenance: SynthesisProvenance | None = None

    def synth(self, text: str, *,
              budget: float | None = None) -> tuple[bytes, int]:
        """Return an empty clip regardless of ``text``."""
        # Silence has no stochastic stage: no seed is used, and the same
        # request always renders the same (empty) clip.
        self.last_provenance = SynthesisProvenance(
            self.model, "none", seed=0, seed_consumed=False, reproducible=True,
            rate_wpm=0)
        return b"", ESPEAK_SAMPLE_RATE

    def cancel(self) -> None:
        """Silence has no process to interrupt."""

    close = cancel


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
    # The espeak-ng voice the mbrola tier runs (mb-us1), resolved at
    # construction; None when the tier is off or has nothing to run.
    _mbrola_voice: str | None = None

    def __init__(self, cfg: dict | None = None, *, voice: str | None = None,
                 rate: int | None = None, mbrola: bool = False,
                 mbrola_fallback: bool = True) -> None:
        self._cfg = cfg or {}
        self.voice = self._checked_voice(
            settings.tts_voice() if voice is None else voice)
        self.rate = int(settings.tts_rate() if rate is None else rate)
        self.mbrola = bool(mbrola)
        self._mbrola_fallback = bool(mbrola_fallback)
        self.model = (models.TTS_ENGINE_MBROLA if self.mbrola
                      else models.TTS_ENGINE_ESPEAK)
        # Set when an mbrola voice turns out not to be installed, so a TUI can
        # explain why the voice sounds like plain espeak.
        self.mbrola_error = ""
        self._mbrola_ok = self.mbrola
        if self.mbrola:
            try:
                self._mbrola_voice = resolve_mbrola_voice(self.voice)
            except TtsError as error:
                # An exact model=mbrola request is refused here, before anything
                # is accepted or launched, naming what is installed. The
                # settings tier speaks plain espeak instead of starting a
                # process that cannot succeed.
                if not self._mbrola_fallback:
                    raise
                self._mbrola_ok = False
                self.mbrola_error = str(error)
        # A13: what produced the last clip. Set only when a synthesis
        # succeeds, so the failed mbrola attempt never leaves it behind.
        self.last_provenance: SynthesisProvenance | None = None

    @property
    def effective_model(self) -> str | None:
        """The family that produced the last clip, for a TUI; None before one.

        Read-only and derived: the daemon carries the full provenance with
        each clip instead of reading this afterwards.
        """
        recorded = self.last_provenance
        return None if recorded is None else recorded.model

    @property
    def selected_voice(self) -> str:
        """The voice a reply names: the MBROLA voice id while that tier is in use
        (us1 for en-us), otherwise the eSpeak voice."""
        if self._mbrola_ok and self._mbrola_voice:
            return self._mbrola_voice[len("mb-"):]
        return self.voice

    @staticmethod
    def _checked_voice(voice: str) -> str:
        token = str(voice).strip()
        if not _VOICE_TOKEN.match(token):
            raise TtsError(
                f"invalid voice name {voice!r}: expected 1-32 characters from "
                "[A-Za-z0-9_+-]. Use a name the engine lists, such as en-us "
                "(run: espeak-ng --voices).")
        return token

    def synth(self, text: str, *,
              budget: float | None = None) -> tuple[bytes, int]:
        """Return (s16le mono PCM, sample rate) for one clip of ``text``."""
        started = time.monotonic()
        clean = _as_text(text).strip()
        if not clean:
            # A screen of nothing but decoration conditions down to nothing;
            # that is an empty clip, not a failure to synthesise.
            return b"", ESPEAK_SAMPLE_RATE
        if self._mbrola_ok and self._mbrola_voice:
            try:
                return self._run(clean, self._mbrola_voice, budget=budget)
            except TtsDeadlineExceeded:
                # A budget that ran out says nothing about whether the mbrola
                # voice is installed. Treating it as a failure marked mbrola
                # broken for the rest of the page and started a fallback
                # process with no budget left to spend on it.
                raise
            except TtsError as error:
                if not self._mbrola_fallback:
                    raise
                # A quality tier that is not installed must never lose a read.
                # Remembering the failure keeps the rest of the page from
                # paying for a doomed process once per sentence.
                self._mbrola_ok = False
                self.mbrola_error = str(error)
        # The fallback inherits what the FIRST attempt left, not a fresh
        # allowance: two full budgets would let one request take twice as long
        # as it asked for.
        left = (None if budget is None
                else budget - (time.monotonic() - started))
        return self._run(clean, self.voice, budget=left)

    def _run(self, text: str, voice: str, *,
             budget: float | None = None) -> tuple[bytes, int]:
        command = build_synth_cmd(self._cfg, voice=voice, rate=self.rate)
        cap = SYNTH_TIMEOUT_BASE_S + len(text) * SYNTH_TIMEOUT_PER_CHAR_S
        timeout = _bounded(cap, budget)
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
        except KeyboardInterrupt:
            # Foreground auditions can be interrupted while communicate is
            # waiting. Reap the child before returning to the next prompt.
            process.kill()
            process.communicate()
            raise
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            failure = TtsDeadlineExceeded if _budget_cut(cap, budget) else TtsError
            raise failure(
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
            clip = util.parse_wav_bytes(out)
        except ValueError as error:
            raise TtsError(
                f"{command[0]} produced no usable audio: {error}"
                f"{_stderr_note(err)} "
                f"{_failure_hint(command[0], voice)}.") from error
        # Only here, after audio was actually produced: the family is decided
        # by the voice this run used, so the mbrola fallback's success records
        # espeak and a failed mb- attempt records nothing.
        self.last_provenance = SynthesisProvenance(
            models.TTS_ENGINE_MBROLA if voice.startswith("mb-")
            else models.TTS_ENGINE_ESPEAK, voice,
            # A14: espeak-ng and mbrola have no stochastic stage, so no seed
            # is used and identical input renders identical audio; the opt-in
            # RealEspeakReproducible test is the evidence on a real engine.
            seed=0, seed_consumed=False, reproducible=True, rate_wpm=self.rate)
        return clip

    def cancel(self) -> None:
        """eSpeak clips are short and have no persistent process to close."""

    close = cancel


def piper_binary() -> str | None:
    """Return the fixed provider command selected by the trusted environment."""
    override = os.environ.get(PIPER_ENV_COMMAND)
    if override:
        return util.which(override)
    managed = os.path.join(paths.data_dir(), "piper", "current", "bin",
                           "kilix-piper-tts")
    return util.which(managed) or util.which("kilix-piper-tts")


def piper_probe(*, budget: float | None = None) -> tuple[bool, str]:
    """Inspect the provider and pinned model without starting it or networking.

    Returns (available, detail) for every provider state, and raises
    TtsDeadlineExceeded only when the caller's budget is what stopped the
    inspection: spent before anything ran, or the bound on a status process
    that timed out. A daemon handler needs that distinction as a type, because
    "the provider is broken" and "we ran out of time asking" otherwise look
    identical -- and conflating them in either direction is R6 finding 1 or
    R5 finding A.
    """
    # The spent budget is checked FIRST, before the PATH lookup: nothing about
    # the install state can change the answer, and checking the binary first
    # made this refusal read "not installed" on any machine without Piper.
    if budget is not None and budget <= 0:
        raise TtsDeadlineExceeded(
            "the request deadline elapsed before the speech provider could "
            "be checked")
    binary = piper_binary()
    if binary is None:
        return False, (
            "kilix-piper-tts is not installed. Install the public "
            "kilix-piper-tts module, then run `kilix-tts --install "
            f"{models.PIPER_KRISTIN_MODEL}`."
        )
    status_timeout = _bounded(PIPER_STATUS_TIMEOUT_S, budget)
    try:
        result = subprocess.run(
            [binary, "status", "--json"],
            capture_output=True,
            check=False,
            timeout=status_timeout,
            text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired as error:
        if _budget_cut(PIPER_STATUS_TIMEOUT_S, budget):
            raise TtsDeadlineExceeded(
                f"the request deadline elapsed while {binary} status was "
                "running") from error
        return False, (f"{binary} status did not answer within "
                       f"{PIPER_STATUS_TIMEOUT_S:g} s")
    except OSError as error:
        return False, f"cannot inspect {binary}: {error}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[0][:200]}" if detail else ""
        return False, f"{binary} status exited {result.returncode}{suffix}"
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return False, f"{binary} status returned invalid JSON: {error}"
    if not isinstance(payload, dict):
        return False, f"{binary} status returned no JSON object"
    if (payload.get("model") != models.PIPER_KRISTIN_MODEL
            or payload.get("voice") != PIPER_VOICE):
        return False, (
            f"{binary} is an incompatible Piper provider: expected "
            f"{models.PIPER_KRISTIN_MODEL}/{PIPER_VOICE}. Reinstall the "
            "matching kilix-piper-tts release."
        )
    installed = payload.get("installed") is True
    detail = str(payload.get("detail") or "provider gave no model detail")
    if installed:
        state = "warm provider" if payload.get("loaded") else "provider starts on demand"
        return True, f"{PIPER_VOICE} · {binary} · {state}"
    return False, detail


def piper_status(*, budget: float | None = None) -> tuple[bool, str]:
    """Inspect the provider; always RETURN (available, detail), never raise.

    The contract kilix-tts and the status page rely on. A caller that must
    tell a deadline from a provider failure calls piper_probe instead.
    """
    try:
        return piper_probe(budget=budget)
    except TtsDeadlineExceeded as error:
        return False, str(error)


class PiperTts:
    """Pinned Kristin synthesis through the isolated persistent provider."""

    name = models.TTS_ENGINE_PIPER
    model = models.PIPER_KRISTIN_MODEL
    voice = PIPER_VOICE
    # A13: set when a synthesis completes or is cancelled; None before one.
    last_provenance: SynthesisProvenance | None = None

    def __init__(self, *, voice: str | None = None,
                 rate: int | None = None) -> None:
        if voice is not None and str(voice).strip().lower() not in {
                "kristin", PIPER_VOICE.lower(), self.model.lower()}:
            raise TtsUnsupported(
                f"model {self.model!r} has the fixed voice {PIPER_VOICE!r}; "
                "omit --voice or use --voice en_US-kristin-medium."
            )
        self.rate = int(settings.tts_rate() if rate is None else rate)
        if self.rate not in (120, 150, 170, 200, 240):
            raise TtsUnsupported(
                "Piper rate must be one of: 120, 150, 170, 200, 240 wpm")
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._cancelled = False

    def _provenance(self) -> SynthesisProvenance:
        """A13/A14 for a Piper clip.

        The model and voice are fixed. kilix-piper-tts sets only the length
        scale: no seed reaches the model, and its noise is drawn fresh on every
        synthesis, so identical text and settings give different audio. There
        is no seed to report. This used to report seed 0, so every descriptor
        presented a seed and settings that could not reproduce the clip. The
        clip carries no seed and says it is not reproducible.
        """
        return SynthesisProvenance(self.model, self.voice, seed=None,
                                   seed_consumed=False, reproducible=False,
                                   rate_wpm=self.rate)

    def check_available(self, *, budget: float | None = None) -> None:
        """Raise TtsError with the provider's own detail unless it is ready.

        TtsDeadlineExceeded propagates from the probe unchanged, so a caller
        can tell a budget that cut the check from a provider that failed it.
        """
        available, detail = piper_probe(budget=budget)
        if not available:
            raise TtsError(detail)

    def synth(self, text: str, *,
              budget: float | None = None) -> tuple[bytes, int]:
        clean = _as_text(text).strip()
        if not clean:
            return b"", PIPER_SAMPLE_RATE
        timeout = _bounded(PIPER_SYNTH_TIMEOUT_S, budget)
        binary = piper_binary()
        if binary is None:
            raise TtsError(
                "kilix-piper-tts is not installed. Install that module and run "
                f"`kilix-tts --install {self.model}`."
            )
        command = [
            binary, "synthesize", "--stdin", "--raw", "--model", self.model,
            "--rate", str(self.rate),
        ]
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        except OSError as error:
            raise TtsError(f"cannot run {binary!r}: {error}") from error
        with self._lock:
            self._process = process
            cancelled = self._cancelled
        if cancelled:
            try:
                process.kill()
            except OSError:
                pass
        try:
            out, err = process.communicate(
                clean.encode("utf-8", "replace"), timeout=timeout)
        except KeyboardInterrupt:
            process.kill()
            process.communicate()
            raise
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            failure = (TtsDeadlineExceeded
                       if _budget_cut(PIPER_SYNTH_TIMEOUT_S, budget) else TtsError)
            raise failure(
                f"{binary} did not finish within {timeout:.0f} "
                "seconds; run its status command and retry."
            ) from error
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
        with self._lock:
            cancelled = self._cancelled
        if cancelled:
            # The fixed model and voice were what ran; the clip is empty.
            self.last_provenance = self._provenance()
            return b"", PIPER_SAMPLE_RATE
        if process.returncode != 0:
            raise TtsError(
                f"{binary} exited {process.returncode}{_stderr_note(err)}. "
                f"Run: {binary} status"
            )
        if len(out) % 2:
            raise TtsError(f"{binary} returned an odd-length s16le PCM stream")
        self.last_provenance = self._provenance()
        return out, PIPER_SAMPLE_RATE

    def cancel(self) -> None:
        """Kill the client; the provider observes EOF and drops its worker."""
        with self._lock:
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    close = cancel


def make_tts(cfg: dict | None = None, *, model: str | None = None,
             voice: str | None = None,
             rate: int | None = None) -> NullTts | EspeakTts | PiperTts | QwenProviderTts:
    """Return the selected engine, with optional request-scoped overrides.

    Construction deliberately does not probe for espeak-ng: a missing
    synthesiser has to degrade the read at the moment it is asked for, with a
    message saying how to install it, rather than stop a TUI or the daemon
    from starting at all.
    """
    if model is None:
        engine = settings.tts_engine()
    else:
        try:
            engine = models.tts_engine_for_model(model)
        except KeyError as error:
            raise TtsError(
                f"unknown TTS model {model!r}; choose one of: "
                f"{', '.join(models.TTS_MODEL_IDS)}. Model paths and commands "
                "cannot be supplied by a speak request.") from error
    if engine == "off":
        return NullTts()
    if engine == models.TTS_ENGINE_PIPER:
        return PiperTts(voice=voice, rate=rate)
    if engine == models.TTS_ENGINE_QWEN:
        from .qwen_provider import QwenProviderTts
        return QwenProviderTts(voice=voice, rate=rate)
    # settings.tts_engine() validates against the vocabulary, so anything that
    # is not "off" is espeak, with or without the mbrola tier on top.
    return EspeakTts(
        cfg, voice=voice, rate=rate,
        mbrola=(engine == models.TTS_ENGINE_MBROLA),
        # The longstanding persistent mbrola setting is a preferred quality
        # tier and keeps its eSpeak fallback. An explicit request model is an
        # exact choice: failure must be reported rather than disguised.
        mbrola_fallback=(model is None))


def render_text(text: str, *, model: str | None = None,
                voice: str | None = None, rate: int | None = None,
                max_chars: int | None = None,
                cfg: dict | None = None) -> RenderedSpeech:
    """Synchronously synthesise a whole turn without opening an audio device.

    The same conditioning, chunking and model factory used by the daemon are
    used here.  Clips are concatenated only when their sample rates match; a
    container must never label PCM from two rates as though it had one.
    """
    engine = make_tts(cfg, model=model, voice=voice, rate=rate)
    chunks = speech_chunks(text, max_chars=max_chars)
    rendered: list[bytes] = []
    sample_rate: int | None = None
    try:
        for chunk in chunks:
            pcm, chunk_rate = engine.synth(chunk)
            if sample_rate is None:
                sample_rate = chunk_rate
            elif chunk_rate != sample_rate:
                raise TtsError(
                    f"the synthesiser changed sample rate from {sample_rate} to "
                    f"{chunk_rate} Hz between clips. Save shorter text with one "
                    "voice, or fix the engine so every clip uses one rate.")
            rendered.append(pcm)
        # The file is labelled with what produced it: the mbrola tier that fell
        # back to plain espeak rendered espeak, not mbrola.
        model, voice = engine.model, engine.voice
        recorded = getattr(engine, "last_provenance", None)
        if isinstance(engine, EspeakTts):
            voice = engine.selected_voice
            if isinstance(recorded, SynthesisProvenance):
                model, voice = recorded.model, recorded.voice
                if model == models.TTS_ENGINE_MBROLA and voice.startswith("mb-"):
                    voice = voice[len("mb-"):]
        return RenderedSpeech(
            b"".join(rendered),
            ESPEAK_SAMPLE_RATE if sample_rate is None else sample_rate,
            len(chunks), model, voice, engine.rate)
    finally:
        closer = getattr(engine, "close", None)
        if closer is not None:
            closer()
