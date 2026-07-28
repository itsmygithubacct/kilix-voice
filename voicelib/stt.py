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
import json
import os
import re

from . import paths, settings
from .util import cfg_get

DEFAULT_RATE = 16000
LIBRARY_BASENAME = "libvosk.so"

# Overrides for a library or a model kept outside the Kilix data directory: a
# distribution package, a hand-built library, or a fixture in the test suite.
ENV_LIBRARY = "KILIX_VOICE_LIBVOSK"
ENV_MODEL = "KILIX_VOICE_MODEL_PATH"

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

ENGINE_VOSK = "vosk"
ENGINE_VIBEVOICE = "vibevoice"
ENGINE_OFF = "off"

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


def make_stt(cfg: dict | None = None, rate: int | None = None) -> NullStt | VoskStt:
    """Return the recogniser the shared settings select.

    ``cfg`` may override the settings file for a caller that already knows what
    it wants — ``stt.engine``, ``stt.model``, ``stt.model_path``,
    ``stt.lib_path``, ``audio.rate`` and ``settings_path`` are read.  ``off``
    and any value this release does not implement give a :class:`NullStt`, so a
    missing engine disables dictation instead of blocking a launch.
    """
    config = cfg or {}
    settings_path = cfg_get(config, "settings_path")
    engine = str(cfg_get(config, "stt.engine")
                 or settings.stt_engine(settings_path)).strip().lower()
    if engine == ENGINE_VIBEVOICE:
        # Never fall through to vosk here: a user who selected vibevoice would
        # otherwise be told nothing and believe they were running it.
        raise SttError(
            f"{settings.KEY_STT_ENGINE}={ENGINE_VIBEVOICE} is not implemented "
            "in this release — the VibeVoice recogniser arrives in a later "
            f"phase. Set it to {ENGINE_VOSK!r} for local recognition now, or "
            f"{ENGINE_OFF!r} to disable dictation; kilix-voice will not "
            "quietly run an engine other than the one you chose.")
    if engine != ENGINE_VOSK:
        return NullStt()
    if rate is None:
        rate = int(cfg_get(config, "audio.rate", DEFAULT_RATE))
    return VoskStt(
        rate,
        model_id=cfg_get(config, "stt.model"),
        model_path=cfg_get(config, "stt.model_path"),
        lib_path=cfg_get(config, "stt.lib_path"),
        settings_path=settings_path,
    )
