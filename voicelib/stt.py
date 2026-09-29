"""Speech recognition: a ctypes binding to libvosk, plus its stand-in.

kilix-voice is standard-library only, so recognition is not a wheel — this
module opens ``libvosk.so`` with :mod:`ctypes` and declares the C entry points
it needs itself.  Dictation therefore runs under the system Python with
nothing installed into it, and every failure stays ours: a missing library, a
missing model, or a library that is not vosk becomes an :class:`SttError`
naming the path that was tried, never a bare ``OSError`` from the loader and
never a segmentation fault inside the loaded code.

Engines here are synchronous primitives — one utterance at a time, one frame
in, at most one string out.  Threads, the microphone and the turn timeout
belong to the daemon.  Importing this module loads nothing: the library is
opened when a :class:`VoskStt` is constructed, not before.
"""

from __future__ import annotations

import ctypes
import dataclasses
import json
import os
import re
import select
import stat
import subprocess
import tempfile
import threading
import time
import wave

from typing import NamedTuple

from . import consent, models, paths, protocol, resources, settings
from .util import cfg_get

DEFAULT_RATE = 16000
LIBRARY_BASENAME = "libvosk.so"

# Overrides for a library or a model kept outside the Kilix data directory: a
# distribution package, a hand-built library, or a fixture in the test suite.
ENV_LIBRARY = "KILIX_VOICE_LIBVOSK"
ENV_MODEL = "KILIX_VOICE_MODEL_PATH"
# The VibeASR.cpp `asr_infer` executable. Without the override the managed
# build under the voice data directory is used; there is deliberately no PATH
# fallback, because `asr_infer` is too generic a name to trust from PATH.
ENV_VIBEASR = "KILIX_VOICE_VIBEASR"
VIBEASR_BASENAME = "asr_infer"
VIBEASR_VAE = "vibeasr-vae-encoder-i8_s.gguf"
VIBEASR_LM = "vibeasr-lm-i2_s-embed-q6_k.gguf"
# asr_infer loads both GGUFs (about two seconds) and then transcribes at an RTF
# near 0.7 on four threads. The bound is generous so a loaded machine is slow
# rather than refused, and finite so a wedged process cannot hold the turn.
VIBEASR_BASE_TIMEOUT_S = 30.0
VIBEASR_TIMEOUT_PER_AUDIO_S = 4.0
VIBEASR_MAX_THREADS = 4
# The kilix-whisper-stt provider (faster-whisper in its own pinned
# environment). Like asr_infer there is no PATH fallback: without the override
# only the managed install under the voice data directory is run.
ENV_WHISPER = "KILIX_VOICE_WHISPER"
WHISPER_BASENAME = "kilix-whisper-stt"
# The provider imports its libraries and loads the model (about two seconds
# on the reference laptop) while the user is already speaking. A decode runs
# at an RTF near 0.3 on four threads; the bounds are generous so a loaded
# machine is slow rather than refused, and finite so a wedged child cannot
# hold the turn.
WHISPER_READY_TIMEOUT_S = 60.0
WHISPER_BASE_TIMEOUT_S = 20.0
WHISPER_TIMEOUT_PER_AUDIO_S = 3.0
WHISPER_MAX_THREADS = 4
# A reply is one JSON line holding a transcript; anything longer is not one.
WHISPER_MAX_REPLY_BYTES = 1 << 20
# The daemon bounds a turn by stt.max_seconds; this is the engine's own ceiling
# on buffered PCM (10 minutes at 16 kHz), so a caller bug cannot grow it freely.
VIBEASR_MAX_BUFFER_BYTES = 16000 * 2 * 600

# vosk logs to stderr, which under a curses TUI is the user's screen.
LOG_LEVEL_SILENT = -1

# A capture frame is 640 bytes (16 kHz, 20 ms).  The ceiling exists only so a
# caller that hands over a whole recording cannot overflow the c_int length
# argument and leave the library reading past the end of the buffer.
MAX_FEED_BYTES = 1 << 20

# libvosk accepts any rate its model tolerates; this range only rejects values
# that would be a caller bug (0, negative, or a byte count mistaken for a rate).
MIN_RATE = 4000
MAX_RATE = 192000

# Kept as public aliases for callers that already import the recognizer module.
# The data-only catalog owns the vocabulary from 0.1.3 onward.
ENGINE_VOSK = models.ENGINE_VOSK
ENGINE_VIBEVOICE = models.ENGINE_VIBEVOICE
ENGINE_WHISPER = models.ENGINE_WHISPER
ENGINE_OFF = models.ENGINE_OFF

# name -> (argtypes, restype).  This is the whole of the C surface kilix-voice
# uses; nothing else in the library is called.
_PROTOTYPES: tuple[tuple[str, tuple[type, ...], object], ...] = (
    ("vosk_set_log_level", (ctypes.c_int,), None),
    ("vosk_model_new", (ctypes.c_char_p,), ctypes.c_void_p),
    ("vosk_model_free", (ctypes.c_void_p,), None),
    ("vosk_recognizer_new", (ctypes.c_void_p, ctypes.c_float), ctypes.c_void_p),
    ("vosk_recognizer_free", (ctypes.c_void_p,), None),
    ("vosk_recognizer_accept_waveform",
     (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int), ctypes.c_int),
    ("vosk_recognizer_partial_result", (ctypes.c_void_p,), ctypes.c_char_p),
    ("vosk_recognizer_final_result", (ctypes.c_void_p,), ctypes.c_char_p),
)

# Recognised text is inserted into a PTY, so it is scrubbed here as well as at
# the injection site: control characters below 0x20 and DEL never survive, and
# with them go the escape-sequence introducers and any trailing newline.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class SttError(RuntimeError):
    """Recognition is unavailable or failed; the message says what to do."""

    code = protocol.ERR_UNAVAILABLE


def _clean_text(raw: str) -> str:
    """Return recognised text safe to hand onwards: no controls, no runs."""
    return " ".join(_CONTROL_CHARS.sub(" ", raw).split())


def _frame_bytes(frame: bytes) -> bytes:
    """Return a validated s16le frame, or raise SttError explaining the input."""
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise SttError(
            f"feed() takes s16le PCM bytes, got {type(frame).__name__}. Pass "
            "the frame MicCapture.read() returned.")
    data = frame if isinstance(frame, bytes) else bytes(frame)
    if len(data) > MAX_FEED_BYTES:
        raise SttError(
            f"feed() was given {len(data)} bytes; the limit is "
            f"{MAX_FEED_BYTES}. Feed one capture frame per call — 640 bytes at "
            "16 kHz / 20 ms.")
    # Half a sample would shift every following sample by one byte and turn the
    # rest of the turn into noise.
    return data[:len(data) - (len(data) % 2)]


class NullStt:
    """The recogniser used when dictation is off, and in tests.

    It consumes audio and recognises nothing, so a caller never needs a branch
    for "no engine": it keeps the same utterance state machine as
    :class:`VoskStt`, and rejects the same misuse, so swapping engines cannot
    change the control flow around it.
    """

    name = "null"
    supports_partials = True

    def __init__(self) -> None:
        self._open = False
        self._closed = False

    def start_utterance(self) -> None:
        self._require_live()
        self._open = True

    def feed(self, frame: bytes) -> str | None:
        self._require_live()
        self._require_open()
        _frame_bytes(frame)
        return None

    def end_utterance(self) -> str:
        self._require_live()
        self._open = False
        return ""

    def close(self) -> None:
        self._open = False
        self._closed = True

    def _require_live(self) -> None:
        if self._closed:
            raise SttError(
                "this recogniser has been closed. Build a new one with "
                "make_stt() for the next dictation turn.")

    def _require_open(self) -> None:
        if not self._open:
            raise SttError(
                "feed() was called before start_utterance(). Open every "
                "dictation turn with start_utterance() so audio from one turn "
                "cannot appear in the next.")


def _resolve_library(explicit: str | None = None) -> str:
    """Return the libvosk.so path to load, or raise SttError naming what failed."""
    candidate = explicit or os.environ.get(ENV_LIBRARY) or paths.libvosk_path()
    candidate = os.path.abspath(os.path.expanduser(str(candidate)))
    if os.path.isdir(candidate):
        # paths.lib_dir() and paths.libvosk_path() are both natural things to
        # put in the override, so accept either form.
        candidate = os.path.join(candidate, LIBRARY_BASENAME)
    if not os.path.isfile(candidate):
        raise SttError(
            f"{LIBRARY_BASENAME} was not found at {candidate}. Dictation needs "
            "the vosk library: install it with Kilix's pinned voice installer, "
            f"which writes {os.path.join(paths.lib_dir(), LIBRARY_BASENAME)}, "
            f"or set {ENV_LIBRARY} to a copy you already have. Read-aloud does "
            "not need it.")
    return candidate


def _load_library(path: str) -> ctypes.CDLL:
    """Open ``path`` and declare every prototype on it."""
    try:
        # RTLD_LOCAL keeps the Kaldi and BLAS symbols vosk links statically out
        # of the global namespace, where they could bind into anything else
        # this process loads later.
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        raise SttError(
            f"cannot load {path}: {error}. The file is there but the dynamic "
            "loader refused it — usually an architecture or libc mismatch, or "
            f"a missing dependency (check with: ldd {path}). Install the "
            f"library built for this machine, or point {ENV_LIBRARY} at one."
        ) from error
    for name, argtypes, restype in _PROTOTYPES:
        try:
            function = getattr(lib, name)
        except AttributeError as error:
            raise SttError(
                f"{path} does not export {name}, so it is not a vosk library "
                "(or it predates the API kilix-voice uses). Install the "
                f"version Kilix pins, or point {ENV_LIBRARY} at it.") from error
        function.argtypes = list(argtypes)
        # restype is set even where it is None (void). ctypes otherwise assumes
        # c_int, which silently truncates a returned pointer to 32 bits on a
        # 64-bit build — the handle still looks plausible and the first call
        # that dereferences it crashes somewhere else entirely.
        function.restype = restype
    return lib


def _intended_model_dir(model_id: str | None = None,
                        model_path: str | None = None,
                        settings_path: str | None = None) -> str | None:
    """Where the recogniser WOULD look, with no check that it is there.

    Split out of ``_resolve_model`` for F03.  Consent needs the identity a turn
    is about to use, and it needs it even when the artefact is missing -- the
    named refusal for "not installed" belongs to model loading, and the named
    refusal for "not consented" belongs to the consent gate.  Folding the
    existence check into resolution let a missing directory pre-empt the
    consent gate, which changes WHICH check refuses.  The microphone stays shut
    either way, but a refusal should say the true reason.

    Returns None when even the catalogue lookup fails; construction then raises
    with the specific message it already had.
    """
    candidate = model_path or os.environ.get(ENV_MODEL)
    if not candidate:
        catalog_id = model_id or settings.stt_model(settings_path)
        try:
            candidate = paths.model_dir(catalog_id)
        except paths.PathError:
            return None
    return os.path.abspath(os.path.expanduser(str(candidate)))


def _resolve_model(model_id: str | None = None, model_path: str | None = None,
                   settings_path: str | None = None) -> str:
    """Return the model directory to open, or raise SttError naming what failed."""
    candidate = model_path or os.environ.get(ENV_MODEL)
    if not candidate:
        catalog_id = model_id or settings.stt_model(settings_path)
        try:
            candidate = paths.model_dir(catalog_id)
        except paths.PathError as error:
            raise SttError(str(error)) from error
    target = os.path.abspath(os.path.expanduser(str(candidate)))
    # Checked here rather than left to the library: Kaldi treats an unreadable
    # model as a fatal error and aborts the process instead of returning NULL,
    # so the common case has to be caught before the call is made.  What is
    # *inside* the directory is the library's business — model layouts differ
    # between the small and lgraph builds.
    if not os.path.isdir(target):
        raise SttError(
            f"the speech model directory {target} does not exist. Download the "
            "model with Kilix's voice installer, or set "
            f"{ENV_MODEL} to a vosk model directory you already have. "
            "Read-aloud does not need it.")
    return target


class VoskStt:
    """Offline recognition through libvosk, bound directly with ctypes.

    One instance owns one model and one recogniser and is not thread-safe: the
    daemon feeds it from a single turn at a time.  ``close()`` releases both
    handles and may be called as often as the caller likes.
    """

    name = "vosk"
    supports_partials = True
    # A turn may end on trailing silence only once this engine has recognised
    # a word: the capture's own start-up transient (a pop well above speech
    # level, about a second long on the reference laptop) otherwise opens and
    # closes a VAD segment before anyone speaks, ending the turn empty.
    ends_on_words = True

    def __init__(self, rate: int = DEFAULT_RATE, *, model_id: str | None = None,
                 model_path: str | None = None, lib_path: str | None = None,
                 settings_path: str | None = None,
                 log_level: int = LOG_LEVEL_SILENT) -> None:
        try:
            self._rate = int(rate)
        except (TypeError, ValueError) as error:
            raise SttError(
                f"sample rate must be an integer, got {rate!r}. Capture and "
                f"recognition both run at {DEFAULT_RATE} Hz.") from error
        if not MIN_RATE <= self._rate <= MAX_RATE:
            raise SttError(
                f"sample rate {self._rate} Hz is outside the usable range "
                f"{MIN_RATE}-{MAX_RATE}. Create the recogniser with the rate "
                f"the capture runs at, normally {DEFAULT_RATE}.")

        self._lib_path = _resolve_library(lib_path)
        self._model_path = _resolve_model(model_id, model_path, settings_path)
        self._lib = _load_library(self._lib_path)
        self._model: int | None = None
        self._rec: int | None = None
        self._closed = False
        self._open = False
        self._dirty = False          # audio has reached the recogniser
        self._segments: list[str] = []
        self._partial = ""
        self._last = ""

        self._lib.vosk_set_log_level(int(log_level))
        # os.fsencode, not .encode("utf-8"): the model lives under a path the
        # user chose, which need not be valid UTF-8.
        model = self._lib.vosk_model_new(os.fsencode(self._model_path))
        if not model:
            raise SttError(
                f"libvosk could not open the model at {self._model_path}. The "
                "directory exists but is not a usable vosk model — it is "
                "commonly the archive's outer folder rather than the model "
                "itself, or an interrupted download. Re-fetch the model with "
                f"Kilix's voice installer, or point {ENV_MODEL} at the "
                "directory that directly contains am/ and conf/.")
        self._model = model
        recogniser = self._lib.vosk_recognizer_new(
            model, ctypes.c_float(float(self._rate)))
        if not recogniser:
            # Nothing else has been handed out yet, so the model is ours to
            # free before the exception leaves the constructor.
            self._lib.vosk_model_free(model)
            self._model = None
            raise SttError(
                f"libvosk could not create a recogniser at {self._rate} Hz for "
                f"the model at {self._model_path}. Check that the model "
                "matches the capture rate — the English models Kilix ships are "
                f"{DEFAULT_RATE} Hz — and that the machine has memory free.")
        self._rec = recogniser

    @property
    def rate(self) -> int:
        """Return the sample rate this recogniser was created for."""
        return self._rate

    @property
    def lib_path(self) -> str:
        """Return the libvosk.so that was loaded."""
        return self._lib_path

    @property
    def model_path(self) -> str:
        """Return the model directory that was opened."""
        return self._model_path

    def start_utterance(self) -> None:
        """Begin a turn, discarding anything an abandoned turn left behind."""
        self._require_live()
        if self._dirty:
            # An earlier turn ended without end_utterance(): drain the library's
            # buffers so its audio cannot surface in this turn's text.
            # final_result() is also vosk's reset, which is why the binding
            # needs no separate reset entry point.
            self._result(self._lib.vosk_recognizer_final_result, "text")
            self._dirty = False
        self._segments = []
        self._partial = ""
        self._last = ""
        self._open = True

    def feed(self, frame: bytes) -> str | None:
        """Feed one frame; return the turn's text when it changed, else None."""
        self._require_live()
        if not self._open:
            raise SttError(
                "feed() was called before start_utterance(). Open every "
                "dictation turn with start_utterance() so audio from one turn "
                "cannot appear in the next.")
        data = _frame_bytes(frame)
        if not data:
            return None
        self._dirty = True
        status = self._lib.vosk_recognizer_accept_waveform(
            self._rec, data, len(data))
        if status < 0:
            raise SttError(
                f"libvosk rejected an audio frame (accept_waveform returned "
                f"{status}). Feed signed 16-bit little-endian mono PCM at "
                f"{self._rate} Hz, the rate this recogniser was created with.")
        if status:
            # An endpoint: the segment that just closed is final, and the
            # library's partial buffer restarts empty.
            segment = self._result(self._lib.vosk_recognizer_final_result, "text")
            self._partial = ""
            if segment:
                self._segments.append(segment)
        else:
            self._partial = self._result(
                self._lib.vosk_recognizer_partial_result, "partial")
        rolling = self._rolling()
        if rolling == self._last:
            return None
        self._last = rolling
        return rolling

    def end_utterance(self) -> str:
        """Close the turn and return its full text (never newline-terminated)."""
        self._require_live()
        if not self._open:
            # Safe to call from a caller's finally: an unopened turn has no
            # text, and raising here would mask whatever ended the turn.
            return ""
        segment = self._result(self._lib.vosk_recognizer_final_result, "text")
        self._dirty = False
        self._partial = ""
        if segment:
            self._segments.append(segment)
        text = self._rolling()
        self._segments = []
        self._last = ""
        self._open = False
        return text

    def close(self) -> None:
        """Free the recogniser and the model. Safe to call more than once."""
        self._open = False
        self._closed = True
        # Each handle is cleared before its free() runs, so a second close — or
        # a close racing a constructor that failed halfway — can never hand a
        # freed pointer back to the library.  Order mirrors construction.
        recogniser, self._rec = self._rec, None
        model, self._model = self._model, None
        if recogniser is not None:
            self._lib.vosk_recognizer_free(recogniser)
        if model is not None:
            self._lib.vosk_model_free(model)

    def _require_live(self) -> None:
        if self._closed or self._rec is None:
            raise SttError(
                "this recogniser has been closed. Build a new one with "
                "make_stt() for the next dictation turn.")

    def _rolling(self) -> str:
        """Return the whole turn so far: closed segments plus the partial."""
        return " ".join(part for part in (*self._segments, self._partial) if part)

    def _result(self, function, key: str) -> str:
        """Call a result function and return the cleaned text under ``key``.

        vosk returns a ``const char *`` that it still owns: the buffer belongs
        to the recogniser and the next call into it overwrites the contents.
        Declaring ``restype`` as ``c_char_p`` makes ctypes snapshot those bytes
        into a new Python object at return, so the copy happens here, before
        any other library call can run.  Do not later "simplify" this to
        ``c_void_p`` plus ``ctypes.string_at()`` — that keeps the borrowed
        pointer alive and eventually reads rewritten or freed memory.
        """
        raw = function(self._rec)
        payload = raw.decode("utf-8", "replace") if raw else ""
        if not payload.strip():
            return ""
        try:
            parsed = json.loads(payload)
        except ValueError as error:
            raise SttError(
                f"libvosk returned a result that is not JSON ({error}). Check "
                f"that {self._lib_path} really is libvosk: kilix-voice binds "
                "its entry points by name, and a different library exporting "
                "the same names will return nonsense.") from error
        if not isinstance(parsed, dict):
            raise SttError(
                f"libvosk returned {type(parsed).__name__} where a JSON object "
                f"was expected. Check that {self._lib_path} is the library "
                "version Kilix pins.")
        text = parsed.get(key, "")
        return _clean_text(text) if isinstance(text, str) else ""


def vibeasr_binary() -> str:
    """Return the asr_infer path VibeVoice would run; it may not exist."""
    override = os.environ.get(ENV_VIBEASR)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(paths.data_dir(), "vibeasr", "current", "bin",
                        VIBEASR_BASENAME)


def vibevoice_missing(model_dir: str | None) -> list[str]:
    """Name what VibeVoice dictation lacks; empty when it can run."""
    missing = []
    binary = vibeasr_binary()
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        missing.append(
            f"the VibeASR runtime is not at {binary}; build it with "
            f"`kilix stt --install {models.VIBEVOICE_MODEL}`")
    if not model_dir or not os.path.isdir(model_dir):
        missing.append(f"the {models.VIBEVOICE_MODEL} model is not at "
                       f"{model_dir or paths.models_dir()}")
    else:
        for name in models.REQUIRED_FILES[ENGINE_VIBEVOICE]:
            if not os.path.isfile(os.path.join(model_dir, name)):
                missing.append(f"{name} is missing from {model_dir}")
    return missing


def whisper_binary() -> str:
    """Return the kilix-whisper-stt path Whisper would run; it may not exist."""
    override = os.environ.get(ENV_WHISPER)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(paths.data_dir(), "whisper", "current", "bin",
                        WHISPER_BASENAME)


def whisper_missing(model_dir: str | None) -> list[str]:
    """Name what Whisper dictation lacks; empty when it can run."""
    missing = []
    binary = whisper_binary()
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        missing.append(
            f"the Whisper runtime is not at {binary}; install it with "
            f"`kilix stt --install {models.WHISPER_MODEL}`")
    if not model_dir or not os.path.isdir(model_dir):
        missing.append(f"the {models.WHISPER_MODEL} model is not at "
                       f"{model_dir or paths.model_dir(models.WHISPER_MODEL)}")
    else:
        for name in models.REQUIRED_FILES[ENGINE_WHISPER]:
            if not os.path.isfile(os.path.join(model_dir, name)):
                missing.append(f"{name} is missing from {model_dir}")
    return missing


def _checked_rate(rate) -> int:
    try:
        value = int(rate)
    except (TypeError, ValueError) as error:
        raise SttError(
            f"sample rate must be an integer, got {rate!r}. Capture and "
            f"recognition both run at {DEFAULT_RATE} Hz.") from error
    if not MIN_RATE <= value <= MAX_RATE:
        raise SttError(
            f"sample rate {value} Hz is outside the usable range "
            f"{MIN_RATE}-{MAX_RATE}. Create the recogniser with the rate "
            f"the capture runs at, normally {DEFAULT_RATE}.")
    return value


def _checked_threads(threads, ceiling: int) -> int:
    try:
        return max(1, int(threads) if threads else min(
            ceiling, max(1, (os.cpu_count() or 2) - 1)))
    except (TypeError, ValueError) as error:
        raise SttError(f"stt.threads must be an integer, got {threads!r}.") from error


class _HeldPayload:
    """Consent bound to held model files, for engines that load them later.

    Consent binds bytes, and a child process opens the model after the gate,
    so the model files are opened HERE, right after the consent gate, and
    held. Their digest must equal the one consent was granted for; the child
    is given those same descriptors (``/proc/self/fd/N``), so replacing the
    path afterwards changes nothing it loads; and each held file's stat is
    re-checked around the child's work. Any change re-hashes the held bytes,
    and a transcript is refused unless they are still the consented ones.
    """

    name = ""
    label = ""

    def _hold(self, consented_payload: str | None) -> None:
        self._held: dict[str, tuple[int, str]] = {}
        try:
            for name in models.REQUIRED_FILES[self.name]:
                path = os.path.join(self._model_path, name)
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                except OSError as error:
                    raise SttError(f"cannot open {path}: {error.strerror}.") from error
                self._held[name] = (fd, path)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise SttError(f"{path} is not a regular file.")
            self._stats_taken = time.time_ns()
            self._stats = self._held_stats()
            self._payload = consent.payload_digest_of(self.name, self._held)
            if consented_payload is not None:
                if self._payload != consented_payload:
                    raise SttError(
                        f"the {self.label} model in {self._model_path} changed after "
                        "dictation consent was checked, so it was not loaded. "
                        "Grant consent again with `kilix-stt --grant-consent` if "
                        "the new files are intended.")
        except BaseException:
            self._release()
            raise

    def _held_stats(self) -> dict[str, tuple[int, int, int, int, int]]:
        stats = {}
        for name, (fd, _path) in self._held.items():
            info = os.fstat(fd)
            stats[name] = (info.st_dev, info.st_ino, info.st_size,
                           info.st_mtime_ns, info.st_ctime_ns)
        return stats

    def _settled(self, stats) -> bool:
        """Whether equal stats can be trusted to mean equal bytes.

        A write in the same timestamp tick as the snapshot leaves every stat
        field as it was, so a file whose ctime or mtime was that recent when
        the snapshot was taken is always re-hashed (git's racy-index rule,
        as the consent digest cache uses).
        """
        limit = self._stats_taken - consent.RACY_SECONDS * 1_000_000_000
        return all(mtime < limit and ctime < limit
                   for _dev, _ino, _size, mtime, ctime in stats.values())

    def _require_unchanged(self, when: str) -> None:
        taken = time.time_ns()
        stats = self._held_stats()
        if stats == self._stats and self._settled(stats):
            return
        # Unlinking or replacing the path also moves a held file's ctime, with
        # its bytes intact; only the bytes decide. A rewrite always moves ctime,
        # even one that restores mtime, so it is always re-hashed here.
        if consent.payload_digest_of(self.name, self._held) == self._payload:
            self._stats, self._stats_taken = stats, taken
            return
        raise SttError(
                f"a {self.label} model file was modified {when}, so this turn's "
                "transcript was discarded. Grant consent again with "
                "`kilix-stt --grant-consent` if the change is intended.")

    def _release(self) -> None:
        held, self._held = self._held, {}
        for fd, _path in held.values():
            try:
                os.close(fd)
            except OSError:
                pass


class _WholeUtterance:
    """A turn buffered in full and transcribed at once, with no partials.

    ``provisional()`` lets the daemon ask, at a VAD speech end, whether the
    segment so far holds any words. A start-up pop or a cough transcribes to
    nothing: its audio is dropped and the turn keeps listening. Words are kept
    as the turn's result, and ``end_utterance()`` returns them without a
    second decode when no audio arrived since.
    """

    supports_partials = False
    # Read off the class by the daemon, so a stand-in engine never opts in by
    # accident.
    transcribes_at_end = True

    def _reset_turn(self) -> None:
        self._pcm = bytearray()
        self._provisional: tuple[int, str] | None = None

    def start_utterance(self) -> None:
        self._require_live()
        self._reset_turn()
        self._open = True

    def feed(self, frame: bytes) -> str | None:
        self._require_live()
        if not self._open:
            raise SttError(
                "feed() was called before start_utterance(). Open every "
                "dictation turn with start_utterance() so audio from one turn "
                "cannot appear in the next.")
        data = _frame_bytes(frame)
        if len(self._pcm) + len(data) > VIBEASR_MAX_BUFFER_BYTES:
            raise SttError(
                f"this dictation turn exceeded {self.label}'s ten-minute buffer. "
                f"Lower {settings.KEY_STT_MAX_SECONDS} or dictate in shorter turns.")
        self._pcm += data
        return None

    def provisional(self) -> str:
        """Transcribe the turn so far; an empty result drops that audio."""
        self._require_live()
        if not self._open or not self._pcm:
            return ""
        pcm = bytes(self._pcm)
        text = self._decode(pcm)
        if text:
            self._provisional = (len(pcm), text)
        elif len(self._pcm) == len(pcm):
            self._reset_turn()
        return text

    def end_utterance(self) -> str:
        self._require_live()
        if not self._open:
            return ""
        self._open = False
        pcm, cached = bytes(self._pcm), self._provisional
        self._reset_turn()
        if not pcm:
            return ""
        # feed() only appends, so an equal length is the same audio.
        if cached is not None and cached[0] == len(pcm):
            return cached[1]
        return self._decode(pcm)

    def _require_live(self) -> None:
        if self._closed:
            raise SttError(
                "this recogniser has been closed. Build a new one with "
                "make_stt() for the next dictation turn.")


class VibeVoiceStt(_HeldPayload, _WholeUtterance):
    """VibeVoice-ASR-BitNet through the VibeASR.cpp ``asr_infer`` executable.

    VibeVoice transcribes a whole utterance at once, so this engine buffers the
    turn's PCM and runs one bounded, greedy (deterministic) ``asr_infer`` per
    decode. It has no partial results. The audio goes to a private temporary
    WAV that is removed as soon as the process exits; nothing is kept.
    ``close()`` kills a transcription still running. asr_infer opens the model
    only when it runs, so the files are held from construction
    (:class:`_HeldPayload`).
    """

    name = ENGINE_VIBEVOICE
    label = "VibeVoice"

    def __init__(self, rate: int = DEFAULT_RATE, *, model_path: str | None = None,
                 threads: int | None = None,
                 consented_payload: str | None = None) -> None:
        self._rate = _checked_rate(rate)
        # Checked before the generic directory resolution, whose message is
        # about vosk: say everything VibeVoice lacks, runtime included.
        missing = vibevoice_missing(
            model_path or _intended_model_dir(models.VIBEVOICE_MODEL))
        if missing:
            raise SttError(f"VibeVoice dictation cannot run: {'; '.join(missing)}.")
        self._model_path = _resolve_model(models.VIBEVOICE_MODEL, model_path)
        self._binary = vibeasr_binary()
        self._threads = _checked_threads(threads, VIBEASR_MAX_THREADS)
        self._reset_turn()
        self._open = False
        self._closed = False
        self._process: subprocess.Popen | None = None
        # Guards the closed flag, the held descriptors and the child handle,
        # so close() on another thread can never land between reading the
        # descriptors and starting the child that inherits them.
        self._lifecycle = threading.Lock()
        self._held = {}
        self._hold(consented_payload)

    @property
    def rate(self) -> int:
        return self._rate

    @property
    def model_path(self) -> str:
        return self._model_path

    def _decode(self, pcm: bytes) -> str:
        seconds = len(pcm) / 2 / self._rate
        work = paths.ensure_private_dir(os.path.join(paths.session_dir(), "stt"))
        handle, audio_path = tempfile.mkstemp(prefix="vibevoice-", suffix=".wav", dir=work)
        try:
            with os.fdopen(handle, "wb") as raw, wave.open(raw, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self._rate)
                wav.writeframes(pcm)
            timeout = VIBEASR_BASE_TIMEOUT_S + VIBEASR_TIMEOUT_PER_AUDIO_S * seconds
            with self._lifecycle:
                if self._closed:
                    return ""             # closed before the child started
                vae_fd = self._held[VIBEASR_VAE][0]
                lm_fd = self._held[VIBEASR_LM][0]
                # The child opens the very files consent verified: the held
                # descriptors are passed down and named through /proc/self/fd.
                command = [self._binary,
                           "--vae-model", f"/proc/self/fd/{vae_fd}",
                           "--lm-model", f"/proc/self/fd/{lm_fd}",
                           "--audio", audio_path, "-t", str(self._threads), "--greedy"]
                self._require_unchanged("since dictation consent was checked")
                # A local handle: close() on another thread clears
                # self._process and kills the child, and this thread still
                # reaps it and closes its pipes through communicate().
                process = self._process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, start_new_session=True,
                    pass_fds=(vae_fd, lm_fd))
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()     # reap it and close both pipes
                self._process = None
                raise SttError(
                    f"VibeVoice did not finish transcribing {seconds:.1f} s of "
                    f"audio within {timeout:.0f} s. The machine may be busy; try "
                    "again, or choose a lighter model.") from None
            returncode = process.returncode
            with self._lifecycle:
                self._process = None
                if self._closed:
                    return ""
                self._require_unchanged("while it was being transcribed")
            if returncode != 0:
                detail = _clean_text(stderr.decode("utf-8", "replace"))[-400:]
                raise SttError(
                    f"VibeVoice's asr_infer exited with status {returncode}: "
                    f"{detail or 'no diagnostic'}. Rebuild the runtime with "
                    f"`kilix stt --install {models.VIBEVOICE_MODEL}`.")
            return _clean_text(stdout.decode("utf-8", "replace"))
        finally:
            try:
                os.unlink(audio_path)
            except FileNotFoundError:
                pass

    def close(self) -> None:
        self._open = False
        with self._lifecycle:
            self._closed = True
            self._pcm = bytearray()
            self._kill()
            self._release()

    def _kill(self) -> None:
        # Only kills: the thread in _decode() owns the process and reaps
        # it, so two threads never communicate() with it at once.
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()


class WhisperStt(_HeldPayload, _WholeUtterance):
    """Whisper through one persistent ``kilix-whisper-stt serve`` child.

    The child is started at construction, before the microphone opens, so the
    model loads while the user is already speaking and the first decode costs
    only its own time. It is given the held model descriptors through a
    private directory of ``/proc/self/fd/N`` links, so it loads exactly the
    bytes consent verified (:class:`_HeldPayload`); the held files are
    re-checked once the model has loaded and after every decode. Requests are
    length-prefixed PCM on its stdin, replies one JSON line on its stdout,
    each under a deadline. ``close()`` kills the child; nothing outlives the
    recogniser.
    """

    name = ENGINE_WHISPER
    label = "Whisper"

    def __init__(self, rate: int = DEFAULT_RATE, *, model_path: str | None = None,
                 threads: int | None = None, beam: int | None = None,
                 consented_payload: str | None = None) -> None:
        self._rate = _checked_rate(rate)
        if self._rate != DEFAULT_RATE:
            raise SttError(
                f"Whisper transcribes {DEFAULT_RATE} Hz audio; the capture runs "
                f"at {self._rate} Hz.")
        missing = whisper_missing(
            model_path or _intended_model_dir(models.WHISPER_MODEL))
        if missing:
            raise SttError(f"Whisper dictation cannot run: {'; '.join(missing)}.")
        self._model_path = _resolve_model(models.WHISPER_MODEL, model_path)
        self._binary = whisper_binary()
        self._threads = _checked_threads(threads, WHISPER_MAX_THREADS)
        try:
            self._beam = int(beam) if beam else 5
        except (TypeError, ValueError) as error:
            raise SttError(f"stt.beam must be an integer, got {beam!r}.") from error
        self._reset_turn()
        self._open = False
        self._closed = False
        self._ready = False
        self._busy = False
        self._process: subprocess.Popen | None = None
        self._links: str | None = None
        self._lifecycle = threading.Lock()
        self._held = {}
        self._hold(consented_payload)
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    @property
    def rate(self) -> int:
        return self._rate

    @property
    def model_path(self) -> str:
        return self._model_path

    def _start(self) -> None:
        work = paths.ensure_private_dir(os.path.join(paths.session_dir(), "stt"))
        self._links = tempfile.mkdtemp(prefix="whisper-", dir=work)
        for name, (fd, _path) in self._held.items():
            os.symlink(f"/proc/self/fd/{fd}", os.path.join(self._links, name))
        with self._lifecycle:
            if self._closed:
                return
            self._require_unchanged("since dictation consent was checked")
            self._process = subprocess.Popen(
                [self._binary, "serve", "--model", self._links,
                 "--threads", str(self._threads), "--beam", str(self._beam)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, start_new_session=True,
                pass_fds=tuple(fd for fd, _path in self._held.values()))
            self._stdout = bytearray()

    def _read_reply(self, process: subprocess.Popen, deadline: float, what: str) -> dict:
        fd = process.stdout.fileno()
        while b"\n" not in self._stdout:
            left = deadline - time.monotonic()
            if left <= 0:
                raise SttError(
                    f"Whisper did not answer within the time allowed {what}. The "
                    "machine may be busy; try again, or choose a lighter model.")
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                # End of output comes a moment before the exit status.
                try:
                    code = process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    code = None
                raise SttError(
                    f"the Whisper provider exited{f' with status {code}' if code is not None else ''} "
                    f"{what}. Reinstall it with `kilix stt --install {models.WHISPER_MODEL}`.")
            self._stdout += chunk
            if len(self._stdout) > WHISPER_MAX_REPLY_BYTES:
                raise SttError("the Whisper provider sent an over-long reply.")
        line, _, rest = bytes(self._stdout).partition(b"\n")
        self._stdout = bytearray(rest)
        try:
            reply = json.loads(line)
        except ValueError:
            reply = None
        if not isinstance(reply, dict):
            raise SttError("the Whisper provider sent a reply that is not JSON.")
        if "error" in reply:
            raise SttError(f"Whisper failed {what}: {_clean_text(str(reply['error']))[:400]}.")
        return reply

    def _write_all(self, process: subprocess.Popen, data: bytes, deadline: float) -> None:
        fd = process.stdin.fileno()
        view = memoryview(data)
        while view:
            left = deadline - time.monotonic()
            if left <= 0:
                raise SttError("Whisper did not take the audio within the time allowed.")
            _, ready, _ = select.select([], [fd], [], left)
            if not ready:
                continue
            try:
                written = os.write(fd, view[:65536])
            except BrokenPipeError:
                raise SttError(
                    "the Whisper provider exited while it was being sent audio. "
                    f"Reinstall it with `kilix stt --install {models.WHISPER_MODEL}`.") from None
            view = view[written:]

    def _decode(self, pcm: bytes) -> str:
        seconds = len(pcm) / 2 / self._rate
        with self._lifecycle:
            if self._closed:
                return ""
            if self._process is None:
                raise SttError(
                    "the Whisper provider stopped after an earlier failure; "
                    "the next dictation turn starts a new one.")
            process, self._busy = self._process, True
        try:
            if not self._ready:
                reply = self._read_reply(
                    process, time.monotonic() + WHISPER_READY_TIMEOUT_S,
                    "while loading the model")
                if reply.get("ready") is not True:
                    raise SttError("the Whisper provider did not report ready.")
                self._ready = True
                # Loaded: from here on nothing the path does can reach it,
                # and the bytes it loaded must still be the consented ones.
                self._require_unchanged("while the model was loading")
            deadline = (time.monotonic() + WHISPER_BASE_TIMEOUT_S
                        + WHISPER_TIMEOUT_PER_AUDIO_S * seconds)
            os.set_blocking(process.stdin.fileno(), False)
            self._write_all(process, b'{"pcm_bytes": %d}\n' % len(pcm) + pcm, deadline)
            reply = self._read_reply(process, deadline, f"on {seconds:.1f} s of audio")
            text = reply.get("text")
            if not isinstance(text, str):
                raise SttError("the Whisper provider sent a reply without text.")
            self._require_unchanged("while it was being transcribed")
            return _clean_text(text)
        except SttError:
            if self._closed:
                return ""
            self._stop_child()
            raise
        finally:
            with self._lifecycle:
                self._busy = False
                if self._closed:
                    self._reap(process)

    def _stop_child(self) -> None:
        with self._lifecycle:
            process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            self._reap(process)

    @staticmethod
    def _reap(process: subprocess.Popen) -> None:
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        self._open = False
        with self._lifecycle:
            self._closed = True
            self._pcm = bytearray()
            process, self._process = self._process, None
            if process is not None:
                if process.poll() is None:
                    process.kill()
                # A decode in progress owns the pipes and reaps the child;
                # otherwise nobody else will.
                if not self._busy:
                    self._reap(process)
            self._release()
        links, self._links = self._links, None
        if links:
            for name in os.listdir(links):
                os.unlink(os.path.join(links, name))
            os.rmdir(links)


@dataclasses.dataclass(frozen=True)
class ResolvedStt:
    """The recogniser identity a turn is committed to, resolved exactly once.

    R3 F03: ``_require_capture_consent`` resolved the model and engine its own
    way while ``make_stt`` resolved them another, so nothing established that
    the identity the user consented to was the identity that opened the
    microphone.  A turn now resolves this ONCE and hands the same frozen object
    to both.  ``model_dir`` is the directory that will actually be opened --
    after ``stt.model_path`` and the environment override, not the catalogue
    guess -- so the consent digest binds the artefact the recogniser loads.
    """

    engine: str
    model_id: str
    model_dir: str | None
    settings_path: str | None
    lib_path: str | None
    # S04: where the engine runs and the task it runs, from the catalogue entry
    # of model_id. Decided here, in the one resolution, so the accelerator
    # lease is taken for the same identity consent and construction use. An
    # engine that builds no recogniser, and an id the catalogue does not hold,
    # is a CPU transcription, as every Vosk model is.
    device_class: str = resources.DEVICE_CPU
    task: str = "transcribe"
    # The payload digest the consent gate granted, set by the gate for this
    # turn. An engine that loads its model after construction (VibeVoice)
    # re-verifies the files it holds against exactly this.
    consented_payload: str | None = None


# Engines with one model: selecting the engine selects the model.
_SINGLE_MODEL = {ENGINE_VIBEVOICE: models.VIBEVOICE_MODEL,
                 ENGINE_WHISPER: models.WHISPER_MODEL}


def resolve_stt(cfg: dict | None = None) -> ResolvedStt:
    """Resolve the effective recogniser identity from ``cfg``.

    This is the ONLY place that decides what dictation will run.  It reproduces
    exactly what ``make_stt`` used to decide inline; ``make_stt`` now consumes
    its result rather than deciding again.
    """
    config = cfg or {}
    settings_path = cfg_get(config, "settings_path")
    engine = str(cfg_get(config, "stt.engine")
                 or settings.stt_engine(settings_path)).strip().lower()
    model_id = str(cfg_get(config, "stt.model")
                   or settings.stt_model(settings_path) or "")
    if engine in _SINGLE_MODEL:
        # VibeVoice and Whisper have exactly one model each. Choosing the
        # engine alone (a settings screen may offer only that) must not load
        # the default vosk model's directory into it, so the engine decides
        # the model here.
        model_id = _SINGLE_MODEL[engine]
    model_dir = None
    if engine in (ENGINE_VOSK, *_SINGLE_MODEL):
        # The same lookup the recogniser will make, made once, here -- but
        # WITHOUT the existence check, so that "not installed" is still
        # reported by model loading rather than by consent resolution.
        model_dir = _intended_model_dir(
            model_id=(model_id if engine in _SINGLE_MODEL
                      else cfg_get(config, "stt.model")),
            model_path=cfg_get(config, "stt.model_path"),
            settings_path=settings_path)
    spec = (models.MODEL_BY_ID.get(model_id)
            if engine in (ENGINE_VOSK, *_SINGLE_MODEL) else None)
    return ResolvedStt(
        engine=engine, model_id=model_id, model_dir=model_dir,
        settings_path=settings_path, lib_path=cfg_get(config, "stt.lib_path"),
        device_class=spec.device_class if spec is not None else resources.DEVICE_CPU)


class ConsentIdentity(NamedTuple):
    """What a dictation consent is bound to, for one resolved recogniser."""

    digest: str
    payload_digest: str
    model_id: str
    engine: str
    model_dir: str | None


def consent_identity(resolved: ResolvedStt) -> ConsentIdentity:
    """Return the consent identity of ``resolved`` -- for grant and gate alike.

    R6 finding 7: the daemon's gate hashed the directory the recogniser opens,
    and kilix-stt --grant-consent hashed the catalogue directory, so under a
    model override no grant could ever satisfy the gate. There is now one
    function, and both call it with a configuration resolved the same way.
    """
    payload = consent.payload_digest_at(resolved.model_dir, resolved.engine)
    return ConsentIdentity(
        consent.capture_digest(resolved.model_id, resolved.engine, payload),
        payload, resolved.model_id, resolved.engine, resolved.model_dir)


def make_stt(cfg: dict | None = None, rate: int | None = None, *,
             resolved: ResolvedStt | None = None
             ) -> NullStt | VoskStt | VibeVoiceStt | WhisperStt:
    """Return the recogniser the shared settings select.

    ``cfg`` may override the settings file for a caller that already knows what
    it wants — ``stt.engine``, ``stt.model``, ``stt.model_path``,
    ``stt.lib_path``, ``audio.rate`` and ``settings_path`` are read.  ``off``
    and any value this release does not implement give a :class:`NullStt`, so a
    missing engine disables dictation instead of blocking a launch.
    """
    config = cfg or {}
    # A caller that already resolved the identity (and consented to it) passes
    # it in, so construction cannot pick a different one. Resolving again here
    # is what F03 was.
    target = resolved if resolved is not None else resolve_stt(config)
    if target.engine not in (ENGINE_VOSK, *_SINGLE_MODEL):
        return NullStt()
    if rate is None:
        rate = int(cfg_get(config, "audio.rate", DEFAULT_RATE))
    if target.engine == ENGINE_VIBEVOICE:
        return VibeVoiceStt(rate, model_path=target.model_dir,
                            threads=cfg_get(config, "stt.threads"),
                            consented_payload=target.consented_payload)
    if target.engine == ENGINE_WHISPER:
        return WhisperStt(rate, model_path=target.model_dir,
                          threads=cfg_get(config, "stt.threads"),
                          consented_payload=target.consented_payload)
    return VoskStt(
        rate,
        # model_path, not model_id: the directory is already resolved, so the
        # recogniser opens the exact artefact whose bytes consent hashed.
        # A None here means even the catalogue lookup failed; VoskStt then
        # re-runs the full resolution and raises its own specific message.
        model_path=target.model_dir,
        lib_path=target.lib_path,
        settings_path=target.settings_path,
    )
