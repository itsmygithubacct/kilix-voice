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
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from voicelib import paths, settings, stt

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
    for key in (stt.ENV_LIBRARY, stt.ENV_MODEL):
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

    def test_vibevoice_is_refused_as_a_later_phase(self) -> None:
        with self.assertRaises(stt.SttError) as caught:
            stt.make_stt({"stt": {"engine": stt.ENGINE_VIBEVOICE}})
        message = str(caught.exception)
        self.assertIn(stt.ENGINE_VIBEVOICE, message)
        self.assertIn("later phase", message)
        # A silent fall-back would have failed on the missing library instead.
        self.assertNotIn(stt.LIBRARY_BASENAME, message)

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

    def test_make_stt_refuses_vibevoice_even_where_vosk_works(self) -> None:
        cfg = {"stt": {"engine": stt.ENGINE_VOSK, "lib_path": self.library,
                       "model_path": self.model}}
        recogniser = stt.make_stt(cfg, stt.DEFAULT_RATE)
        self.addCleanup(recogniser.close)
        self.assertIsInstance(recogniser, stt.VoskStt)
        # Same config, same working library: the only difference is the engine
        # name, so a raise here can only mean vibevoice was refused outright
        # rather than quietly served by vosk.
        cfg["stt"]["engine"] = stt.ENGINE_VIBEVOICE
        with self.assertRaises(stt.SttError) as caught:
            stt.make_stt(cfg, stt.DEFAULT_RATE)
        self.assertIn("later phase", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
