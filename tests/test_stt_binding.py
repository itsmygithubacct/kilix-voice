"""The libvosk ctypes binding, proved against a stub library built here.

There is no vosk wheel to import and no model on a test machine, so the binding
is checked two ways.  Its declarations are compared against the seven
prototypes DESIGN.md freezes — ``argtypes`` *and* ``restype`` on every one,
because an undeclared ctypes function quietly defaults to "takes anything,
returns int", which truncates a 64-bit handle to 32 bits and crashes later
somewhere with no connection to the call that broke it.

Then a stub ``libvosk.so`` is compiled from a few lines of C and
:class:`~voicelib.stt.VoskStt` is driven against it end to end, so the calling
convention, the borrowed result pointers and the order of the two ``free``
calls are executed rather than assumed.  The tests that need the compiler skip
where there is none; the failure paths need no compiler and always run.

Nothing here opens an audio device, reads the developer's own settings, or
touches the network: every test runs against a private temporary tree.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from voicelib import consent, models, paths, settings, stt

# One capture frame: 20 ms of 16 kHz s16le mono, 640 bytes. The samples
# themselves are never looked at — the stub decides what it "hears".
FRAME = b"\x20\x00" * 320

# The seven entry points DESIGN.md freezes, with the exact declaration each one
# must carry. A restype of None means the C function returns void.
EXPECTED_PROTOTYPES: tuple[tuple[str, tuple[type, ...], object], ...] = (
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

# A stand-in for libvosk. The result buffer is shared and is deliberately
# scribbled over on every accept_waveform: a binding that kept the borrowed
# const char * instead of copying it at return would read the scribble rather
# than the text, and this is the only place that can be made to happen.
STUB_SOURCE = r"""
#include <stdlib.h>
#include <string.h>

static char result[64];
static int frames;
static int pending;     /* audio has arrived since the last final result */

static const char *hand_out(const char *json)
{
    memset(result, 0, sizeof result);
    strncpy(result, json, sizeof result - 1);
    return result;
}

void vosk_set_log_level(int level) { (void)level; }

void *vosk_model_new(const char *path)
{
    if (path == NULL || path[0] == '\0') return NULL;
    frames = 0;
    pending = 0;
    return calloc(1, 8);
}

void vosk_model_free(void *model) { free(model); }

void *vosk_recognizer_new(void *model, float rate)
{
    if (model == NULL || rate <= 0.0f) return NULL;
    return calloc(1, 8);
}

void vosk_recognizer_free(void *recogniser) { free(recogniser); }

int vosk_recognizer_accept_waveform(void *recogniser, const char *pcm,
                                    int length)
{
    if (recogniser == NULL || pcm == NULL || length <= 0) return -1;
    memset(result, 'X', sizeof result - 1);
    result[sizeof result - 1] = '\0';
    pending = 1;
    frames++;
    return (frames % 3 == 0) ? 1 : 0;   /* an endpoint every third frame */
}

const char *vosk_recognizer_partial_result(void *recogniser)
{
    if (recogniser == NULL) return NULL;
    return hand_out("{\"partial\":\"hello\"}");
}

const char *vosk_recognizer_final_result(void *recogniser)
{
    if (recogniser == NULL) return NULL;
    if (!pending) return hand_out("{\"text\":\"\"}");
    pending = 0;
    /* The newline is escaped inside the JSON on purpose: recognised text is
       typed into a PTY, and dictation must never deliver one. */
    return hand_out("{\"text\":\"hello world\\n\"}");
}
"""

# A valid shared library that exports none of the names the binding needs.
NOT_VOSK_SOURCE = "int kilix_not_vosk(void) { return 0; }\n"


def _compiler() -> str | None:
    """Return a C compiler on PATH, or None when the machine has none."""
    for candidate in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if candidate and shutil.which(candidate):
            return candidate
    return None


def _compile(source: str, directory: str, stem: str) -> str:
    """Build ``source`` into ``<directory>/<stem>.so``; skip with no toolchain."""
    compiler = _compiler()
    if compiler is None:
        raise unittest.SkipTest(
            "no C compiler on PATH (tried $CC, cc, gcc, clang), so the stub "
            "libvosk.so cannot be built and the ctypes binding is only checked "
            "at the declaration level. Install cc/gcc/clang to run these.")
    csource = os.path.join(directory, f"{stem}.c")
    library = os.path.join(directory, f"{stem}.so")
    with open(csource, "w", encoding="utf-8") as handle:
        handle.write(source)
    built = subprocess.run(
        [compiler, "-shared", "-fPIC", "-o", library, csource],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
        check=False)
    if built.returncode != 0:
        raise unittest.SkipTest(
            f"{compiler} could not build the stub library: "
            f"{built.stdout.decode('utf-8', 'replace').strip()}")
    try:
        ctypes.CDLL(library, mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        # A noexec temporary directory builds the library and then refuses to
        # map it, which is the machine's problem rather than the binding's.
        raise unittest.SkipTest(
            f"the dynamic loader refused the freshly built {library} "
            f"({error}). Point TMPDIR at a directory this machine allows code "
            "to be loaded from to run these.") from error
    return library


def _isolate(test: unittest.TestCase) -> str:
    """Point every Kilix path at a private tree; return its root.

    The library, the model and the shared settings file are all resolved from
    the environment, so without this a test would read whatever the developer
    running it happens to have installed.
    """
    root = tempfile.mkdtemp(prefix="kilix-voice-stt-")
    test.addCleanup(shutil.rmtree, root, True)
    patcher = mock.patch.dict(os.environ, {
        "HOME": root,
        "GPU_TERMINAL_HOME": os.path.join(root, "gpu_terminal"),
        "GPU_TERMINAL_SETTINGS_FILE": os.path.join(root, "settings.conf"),
        "KILIX_SESSION_HOME": os.path.join(root, "session"),
        "KILIX_DATA_HOME": os.path.join(root, "data"),
    })
    patcher.start()
    test.addCleanup(patcher.stop)
    # patch.dict restores the whole mapping, so removing overrides here is safe.
    for key in (stt.ENV_LIBRARY, stt.ENV_MODEL, stt.ENV_VIBEASR):
        os.environ.pop(key, None)
    return root


def _declaration(function: object) -> tuple[tuple[type, ...] | None, object]:
    """Return (argtypes, restype) as declared on a loaded ctypes function."""
    argtypes = function.argtypes
    return (None if argtypes is None else tuple(argtypes)), function.restype


def _assert_declarations(test: unittest.TestCase, declared: dict) -> None:
    """Check ``name -> (argtypes, restype)`` against the frozen contract."""
    test.assertEqual(sorted(declared),
                     sorted(name for name, _a, _r in EXPECTED_PROTOTYPES),
                     "the binding must declare exactly the seven entry points "
                     "DESIGN.md freezes")
    for name, argtypes, restype in EXPECTED_PROTOTYPES:
        with test.subTest(prototype=name):
            actual_args, actual_restype = declared[name]
            # An undeclared ctypes function has argtypes None and restype
            # c_int, so both halves are asserted: neither may be left to
            # default, and a void function is proved by restype being None.
            test.assertIsNotNone(actual_args, f"{name} declares no argtypes")
            test.assertEqual(actual_args, argtypes)
            test.assertIs(actual_restype, restype)


class PrototypeDeclarationTestCase(unittest.TestCase):

    def test_the_prototype_table_matches_the_contract(self) -> None:
        declared = {name: (tuple(argtypes), restype)
                    for name, argtypes, restype in stt._PROTOTYPES}
        self.assertEqual(len(stt._PROTOTYPES), len(EXPECTED_PROTOTYPES))
        _assert_declarations(self, declared)

    def test_results_are_declared_as_char_pointers(self) -> None:
        # c_char_p is what makes ctypes copy the borrowed buffer at return;
        # c_void_p plus a later string_at() would keep vosk's own pointer alive
        # and eventually read a rewritten or freed buffer.
        declared = {name: restype for name, _args, restype in stt._PROTOTYPES}
        self.assertIs(declared["vosk_recognizer_partial_result"],
                      ctypes.c_char_p)
        self.assertIs(declared["vosk_recognizer_final_result"], ctypes.c_char_p)


class LibraryResolutionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.root = _isolate(self)

    def test_missing_library_names_the_path_it_tried(self) -> None:
        expected = paths.libvosk_path()
        with self.assertRaises(stt.SttError) as caught:
            stt.VoskStt(stt.DEFAULT_RATE)
        error = caught.exception
        self.assertNotIsInstance(error, OSError)
        self.assertIn(expected, str(error))
        self.assertIn(stt.ENV_LIBRARY, str(error))

    def test_missing_model_names_the_path_it_tried(self) -> None:
        # The library is only resolved — a file has to exist — before the model
        # is looked for, so this failure needs no real library.
        library = os.path.join(self.root, stt.LIBRARY_BASENAME)
        pathlib.Path(library).write_bytes(b"not a library\n")
        expected = paths.model_dir(settings.stt_model())
        with self.assertRaises(stt.SttError) as caught:
            stt.VoskStt(stt.DEFAULT_RATE, lib_path=library)
        error = caught.exception
        self.assertNotIsInstance(error, OSError)
        self.assertIn(expected, str(error))
        self.assertIn(stt.ENV_MODEL, str(error))

    def test_a_library_the_loader_refuses_is_reported_not_raised(self) -> None:
        library = os.path.join(self.root, stt.LIBRARY_BASENAME)
        pathlib.Path(library).write_bytes(b"not a library\n")
        model = os.path.join(self.root, "model")
        os.mkdir(model)
        with self.assertRaises(stt.SttError) as caught:
            stt.VoskStt(stt.DEFAULT_RATE, lib_path=library, model_path=model)
        error = caught.exception
        self.assertNotIsInstance(error, OSError)
        self.assertIn(library, str(error))

    def test_an_invalid_rate_is_refused_before_anything_is_loaded(self) -> None:
        with self.assertRaises(stt.SttError) as caught:
            stt.VoskStt(0)
        self.assertIn(str(stt.DEFAULT_RATE), str(caught.exception))


class EngineSelectionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.root = _isolate(self)

    def test_vibevoice_without_its_runtime_names_the_install(self) -> None:
        with self.assertRaises(stt.SttError) as caught:
            stt.make_stt({"stt": {"engine": stt.ENGINE_VIBEVOICE}})
        message = str(caught.exception)
        self.assertIn("VibeASR runtime", message)
        self.assertIn(f"kilix stt --install {models.VIBEVOICE_MODEL}", message)
        # A silent fall-back would have failed on the missing library instead.
        self.assertNotIn(stt.LIBRARY_BASENAME, message)

    def test_vibevoice_never_resolves_a_vosk_model(self) -> None:
        resolved = stt.resolve_stt({"stt": {"engine": stt.ENGINE_VIBEVOICE,
                                            "model": "small-en-us"}})
        self.assertEqual(resolved.model_id, models.VIBEVOICE_MODEL)
        self.assertEqual(resolved.model_dir, paths.model_dir(models.VIBEVOICE_MODEL))

    def test_vibevoice_in_the_shared_settings_is_refused_too(self) -> None:
        document = pathlib.Path(paths.settings_file())
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(
            f"{settings.KEY_STT_ENGINE}={stt.ENGINE_VIBEVOICE}\n",
            encoding="utf-8")
        with self.assertRaises(stt.SttError) as caught:
            stt.make_stt()
        self.assertIn(stt.ENGINE_VIBEVOICE, str(caught.exception))

    def test_off_and_unknown_engines_give_the_null_recogniser(self) -> None:
        for engine in (stt.ENGINE_OFF, "null", "whisper"):
            with self.subTest(engine=engine):
                recogniser = stt.make_stt({"stt": {"engine": engine}})
                self.addCleanup(recogniser.close)
                self.assertIsInstance(recogniser, stt.NullStt)

    def test_the_null_recogniser_keeps_the_utterance_state_machine(self) -> None:
        recogniser = stt.NullStt()
        with self.assertRaises(stt.SttError):
            recogniser.feed(FRAME)
        recogniser.start_utterance()
        self.assertIsNone(recogniser.feed(FRAME))
        self.assertEqual(recogniser.end_utterance(), "")
        recogniser.close()
        with self.assertRaises(stt.SttError):
            recogniser.start_utterance()


class StubLibraryTestCase(unittest.TestCase):
    """VoskStt driven end to end against a compiled stub libvosk.so.

    The stub answers every partial with ``{"partial":"hello"}``, closes a
    segment with ``{"text":"hello world\\n"}`` on every third frame, and
    returns an empty final result when no audio has arrived since the last one
    — which is how vosk itself behaves once a segment has just been closed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.build_dir = tempfile.mkdtemp(prefix="kilix-voice-stub-")
        cls.addClassCleanup(shutil.rmtree, cls.build_dir, True)
        cls.library = _compile(STUB_SOURCE, cls.build_dir, "libvosk")
        cls.model = os.path.join(cls.build_dir, "model")
        os.makedirs(cls.model, exist_ok=True)

    def setUp(self) -> None:
        _isolate(self)

    def engine(self) -> stt.VoskStt:
        """Return a recogniser bound to the stub, closed when the test ends."""
        recogniser = stt.VoskStt(stt.DEFAULT_RATE, lib_path=self.library,
                                 model_path=self.model)
        self.addCleanup(recogniser.close)
        return recogniser

    def test_loaded_functions_declare_argtypes_and_restype(self) -> None:
        library = stt._load_library(self.library)
        declared = {name: _declaration(getattr(library, name))
                    for name, _args, _restype in stt._PROTOTYPES}
        _assert_declarations(self, declared)

    def test_a_recogniser_reports_where_it_came_from(self) -> None:
        recogniser = self.engine()
        self.assertEqual(recogniser.name, "vosk")
        self.assertTrue(recogniser.supports_partials)
        self.assertEqual(recogniser.rate, stt.DEFAULT_RATE)
        self.assertEqual(recogniser.lib_path, self.library)
        self.assertEqual(recogniser.model_path, self.model)

    def test_one_utterance_from_start_to_final_text(self) -> None:
        recogniser = self.engine()
        recogniser.start_utterance()
        # Frame 1 produces the first partial, frame 2 repeats it (nothing
        # changed, so nothing is reported), frame 3 closes the segment.
        self.assertEqual(recogniser.feed(FRAME), "hello")
        self.assertIsNone(recogniser.feed(FRAME))
        self.assertEqual(recogniser.feed(FRAME), "hello world")
        # The stub's canned text ends with a newline; dictation never delivers
        # one (DESIGN.md safety rule 2).
        self.assertEqual(recogniser.end_utterance(), "hello world")

    def test_a_partial_is_replaced_by_the_final_text(self) -> None:
        recogniser = self.engine()
        recogniser.start_utterance()
        self.assertEqual(recogniser.feed(FRAME), "hello")
        self.assertEqual(recogniser.end_utterance(), "hello world")

    def test_a_new_turn_cannot_inherit_the_previous_one(self) -> None:
        recogniser = self.engine()
        recogniser.start_utterance()
        recogniser.feed(FRAME)
        # Abandoned without end_utterance(): the next turn must drain whatever
        # the library still holds rather than prefix it to the new text.
        recogniser.start_utterance()
        self.assertEqual(recogniser.feed(FRAME), "hello")
        self.assertEqual(recogniser.end_utterance(), "hello world")

    def test_feed_before_start_utterance_is_refused(self) -> None:
        recogniser = self.engine()
        with self.assertRaises(stt.SttError) as caught:
            recogniser.feed(FRAME)
        self.assertIn("start_utterance", str(caught.exception))

    def test_end_utterance_is_safe_without_a_turn(self) -> None:
        self.assertEqual(self.engine().end_utterance(), "")

    def test_close_is_idempotent_and_final(self) -> None:
        recogniser = self.engine()
        recogniser.start_utterance()
        recogniser.feed(FRAME)
        recogniser.close()
        # A second close must free nothing: the stub frees for real, so a
        # handle handed back twice would abort this process.
        recogniser.close()
        for call in (recogniser.start_utterance,
                     lambda: recogniser.feed(FRAME),
                     recogniser.end_utterance):
            with self.subTest(call=call):
                with self.assertRaises(stt.SttError):
                    call()

    def test_the_library_override_may_name_the_directory(self) -> None:
        recogniser = stt.VoskStt(stt.DEFAULT_RATE, lib_path=self.build_dir,
                                 model_path=self.model)
        self.addCleanup(recogniser.close)
        self.assertEqual(recogniser.lib_path, self.library)

    def test_a_library_without_the_vosk_symbols_is_named_as_such(self) -> None:
        other = _compile(NOT_VOSK_SOURCE, self.build_dir, "notvosk")
        with self.assertRaises(stt.SttError) as caught:
            stt.VoskStt(stt.DEFAULT_RATE, lib_path=other,
                        model_path=self.model)
        message = str(caught.exception)
        self.assertNotIsInstance(caught.exception, OSError)
        self.assertIn(other, message)
        self.assertIn("vosk_set_log_level", message)

    def test_make_stt_never_serves_vibevoice_with_vosk(self) -> None:
        cfg = {"stt": {"engine": stt.ENGINE_VOSK, "lib_path": self.library,
                       "model_path": self.model}}
        recogniser = stt.make_stt(cfg, stt.DEFAULT_RATE)
        self.addCleanup(recogniser.close)
        self.assertIsInstance(recogniser, stt.VoskStt)
        # Same config, same working library: the only difference is the engine
        # name, so a raise here can only mean vibevoice was refused for its own
        # missing runtime rather than quietly served by vosk.
        cfg["stt"]["engine"] = stt.ENGINE_VIBEVOICE
        with self.assertRaises(stt.SttError) as caught:
            stt.make_stt(cfg, stt.DEFAULT_RATE)
        self.assertIn("VibeASR runtime", str(caught.exception))


# A stand-in asr_infer: records its argv and the WAV it was given, then prints
# what FAKE_TEXT says (with a control character the engine must strip), exits
# with FAKE_STATUS, or sleeps FAKE_SLEEP seconds first.
FAKE_ASR = r"""#!{python}
import hashlib, json, os, sys, time, wave
args = sys.argv[1:]
audio = args[args.index("--audio") + 1]
# What the runtime would load: the bytes behind each model argument.
loaded = {{flag: hashlib.sha256(open(args[args.index(flag) + 1], "rb").read()).hexdigest()
          for flag in ("--vae-model", "--lm-model")}}
with wave.open(audio) as w:
    shape = [w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({{"argv": args, "wav": shape, "loaded": loaded}}) + "\n")
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
print("loading...", file=sys.stderr)
sys.stdout.write(os.environ.get("FAKE_TEXT", "\nhello\x1b[31m world.\n"))
sys.exit(int(os.environ.get("FAKE_STATUS", "0")))
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class VibeVoiceEngineTestCase(unittest.TestCase):
    """The VibeVoice engine against a fake asr_infer; no weights, no audio."""

    def setUp(self) -> None:
        self.root = _isolate(self)
        self.model = os.path.join(paths.data_dir(), "models", models.VIBEVOICE_MODEL)
        os.makedirs(self.model)
        pathlib.Path(self.model, stt.VIBEASR_VAE).write_bytes(b"vae A")
        pathlib.Path(self.model, stt.VIBEASR_LM).write_bytes(b"lm A")
        self.binary = os.path.join(self.root, "asr_infer")
        pathlib.Path(self.binary).write_text(
            FAKE_ASR.format(python=sys.executable), encoding="utf-8")
        os.chmod(self.binary, 0o755)
        self.log = os.path.join(self.root, "asr.log")
        patcher = mock.patch.dict(os.environ, {stt.ENV_VIBEASR: self.binary,
                                               "FAKE_LOG": self.log})
        patcher.start()
        self.addCleanup(patcher.stop)

    def engine(self, consented_payload=None, **cfg) -> "stt.VibeVoiceStt":
        config = {"stt": {"engine": stt.ENGINE_VIBEVOICE, **cfg}}
        resolved = dataclasses.replace(stt.resolve_stt(config),
                                       consented_payload=consented_payload)
        recogniser = stt.make_stt(config, stt.DEFAULT_RATE, resolved=resolved)
        self.addCleanup(recogniser.close)
        self.assertIsInstance(recogniser, stt.VibeVoiceStt)
        return recogniser

    def calls(self) -> list[dict]:
        if not os.path.exists(self.log):
            return []
        with open(self.log, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    def test_one_turn_is_one_greedy_transcription_of_its_audio(self) -> None:
        recogniser = self.engine(threads=2)
        self.assertFalse(recogniser.supports_partials)
        recogniser.start_utterance()
        for _ in range(50):
            self.assertIsNone(recogniser.feed(FRAME))
        self.assertEqual(recogniser.end_utterance(), "hello [31m world.")
        (call,) = self.calls()
        argv = call["argv"]
        # The runtime is handed the held files, and loads the model's bytes.
        self.assertRegex(argv[argv.index("--vae-model") + 1], r"^/proc/self/fd/\d+$")
        self.assertRegex(argv[argv.index("--lm-model") + 1], r"^/proc/self/fd/\d+$")
        self.assertEqual(call["loaded"], {"--vae-model": sha(b"vae A"), "--lm-model": sha(b"lm A")})
        self.assertIn("--greedy", argv)
        self.assertEqual(argv[argv.index("-t") + 1], "2")
        # mono, 16-bit, the capture rate, and exactly the frames fed
        self.assertEqual(call["wav"], [1, 2, stt.DEFAULT_RATE, 50 * len(FRAME) // 2])
        # the private WAV is gone once the turn is over
        self.assertFalse(os.path.exists(argv[argv.index("--audio") + 1]))

    def test_a_silent_turn_runs_nothing(self) -> None:
        recogniser = self.engine()
        recogniser.start_utterance()
        self.assertEqual(recogniser.end_utterance(), "")
        self.assertEqual(self.calls(), [])

    def test_a_failed_transcription_names_the_status(self) -> None:
        recogniser = self.engine()
        recogniser.start_utterance()
        recogniser.feed(FRAME)
        with mock.patch.dict(os.environ, {"FAKE_STATUS": "7"}):
            with self.assertRaises(stt.SttError) as caught:
                recogniser.end_utterance()
        self.assertIn("status 7", str(caught.exception))
        self.assertIn("loading", str(caught.exception))

    def test_a_wedged_transcription_is_killed_at_its_bound(self) -> None:
        recogniser = self.engine()
        recogniser.start_utterance()
        recogniser.feed(FRAME)
        started = time.monotonic()
        with mock.patch.dict(os.environ, {"FAKE_SLEEP": "30"}), \
                mock.patch.object(stt, "VIBEASR_BASE_TIMEOUT_S", 0.5):
            with self.assertRaises(stt.SttError) as caught:
                recogniser.end_utterance()
        self.assertIn("did not finish", str(caught.exception))
        self.assertLess(time.monotonic() - started, 5)

    def test_turns_do_not_share_audio(self) -> None:
        recogniser = self.engine()
        recogniser.start_utterance()
        recogniser.feed(FRAME * 3)
        recogniser.end_utterance()
        recogniser.start_utterance()
        recogniser.feed(FRAME)
        recogniser.end_utterance()
        # An abandoned turn (started, fed, never ended) leaves nothing behind.
        recogniser.start_utterance()
        recogniser.feed(FRAME * 5)
        recogniser.start_utterance()
        recogniser.feed(FRAME * 2)
        recogniser.end_utterance()
        self.assertEqual([call["wav"][3] for call in self.calls()],
                         [3 * len(FRAME) // 2, len(FRAME) // 2, 2 * len(FRAME) // 2])

    def consented(self) -> str:
        return consent.payload_digest_at(self.model, models.ENGINE_VIBEVOICE)

    def turn(self, recogniser) -> str:
        recogniser.start_utterance()
        recogniser.feed(FRAME)
        return recogniser.end_utterance()

    def test_a_path_swap_after_consent_cannot_change_what_loads(self) -> None:
        # Seat 1's High: consent is checked, then the model is replaced at the
        # same path while the turn records. The runtime must still load A.
        recogniser = self.engine(consented_payload=self.consented())
        swap = os.path.join(self.model, stt.VIBEASR_LM + ".new")
        pathlib.Path(swap).write_bytes(b"lm B")
        os.replace(swap, os.path.join(self.model, stt.VIBEASR_LM))
        self.assertEqual(self.turn(recogniser), "hello [31m world.")
        (call,) = self.calls()
        self.assertEqual(call["loaded"]["--lm-model"], sha(b"lm A"))

    def test_an_in_place_rewrite_during_the_turn_discards_it(self) -> None:
        recogniser = self.engine(consented_payload=self.consented())
        time.sleep(0.05)                 # a later timestamp tick than the write
        with open(os.path.join(self.model, stt.VIBEASR_LM), "r+b") as handle:
            handle.write(b"lm B")        # same inode, same length
        with self.assertRaises(stt.SttError) as caught:
            self.turn(recogniser)
        self.assertIn("modified", str(caught.exception))
        self.assertEqual(self.calls(), [])

    def test_a_rewrite_that_restores_mtime_is_still_seen(self) -> None:
        recogniser = self.engine(consented_payload=self.consented())
        target = os.path.join(self.model, stt.VIBEASR_LM)
        before = os.stat(target)
        time.sleep(0.05)
        with open(target, "r+b") as handle:
            handle.write(b"lm B")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaises(stt.SttError):
            self.turn(recogniser)
        self.assertEqual(self.calls(), [])

    def test_a_model_swapped_before_the_engine_opens_it_is_refused(self) -> None:
        granted = self.consented()
        pathlib.Path(self.model, stt.VIBEASR_LM).write_bytes(b"lm B")
        with self.assertRaises(stt.SttError) as caught:
            self.engine(consented_payload=granted)
        self.assertIn("changed after dictation consent", str(caught.exception))

    def test_close_releases_the_held_files(self) -> None:
        recogniser = stt.VibeVoiceStt(stt.DEFAULT_RATE, model_path=self.model)
        held = [fd for fd, _path in recogniser._held.values()]
        recogniser.close()
        for fd in held:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_a_missing_gguf_is_named(self) -> None:
        os.unlink(os.path.join(self.model, stt.VIBEASR_LM))
        with self.assertRaises(stt.SttError) as caught:
            stt.make_stt({"stt": {"engine": stt.ENGINE_VIBEVOICE}}, stt.DEFAULT_RATE)
        self.assertIn(stt.VIBEASR_LM, str(caught.exception))

    def test_the_consent_identity_binds_the_vibevoice_payload(self) -> None:
        resolved = stt.resolve_stt({"stt": {"engine": stt.ENGINE_VIBEVOICE}})
        self.assertEqual(resolved.model_id, models.VIBEVOICE_MODEL)
        self.assertEqual(resolved.model_dir, self.model)
        self.assertTrue(stt.consent_identity(resolved).payload_digest)


if __name__ == "__main__":
    unittest.main()
