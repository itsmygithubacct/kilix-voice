"""MB-01: model=mbrola runs an installed MBROLA voice, or refuses naming them.

The mbrola tier ran mb-<voice> verbatim. The shared default voice is en-us,
and espeak-ng has no voice called mb-en-us, so model=mbrola failed as shipped,
even on a machine with a US English MBROLA voice installed. A language now
resolves to the most preferred of its MBROLA voices whose database espeak-ng
would find. An explicit MBROLA voice id is used as named. An exact model=mbrola
request with nothing installed for the language is refused before anything
runs, and the refusal names the MBROLA voices that are installed. The settings
tier then speaks plain espeak instead of starting a process that cannot
succeed. Replies, status and rendered files name the voice that runs.

Every test here decides which databases are "installed" through XDG_DATA_DIRS,
the search path espeak-ng itself uses, so nothing depends on the host.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_mbrola", importlib.machinery.SourceFileLoader(
        "kilix_voiced_mbrola", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol, settings, tts  # noqa: E402

SYNTHESISER = r'''
import json, struct, sys
voice, log, mode = sys.argv[1:4]
sys.stdin.buffer.read()
with open(log, "a") as fh:
    fh.write(json.dumps(voice) + "\n")
if mode == "mb-fails" and voice.startswith("mb-"):
    sys.stderr.write("mbrola voice %s failed\n" % voice)
    sys.exit(1)
pcm = b"\x01\x00" * 400
fmt = struct.pack("<HHIIHH", 1, 1, 22050, 44100, 2, 16)
body = (b"fmt " + struct.pack("<I", 16) + fmt + b"data"
        + struct.pack("<I", len(pcm)) + pcm)
sys.stdout.buffer.write(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
'''


class _MbrolaFixture(unittest.TestCase):

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = directory.name
        self.share = os.path.join(self.root, "share")
        os.makedirs(os.path.join(self.share, "mbrola"))
        script = os.path.join(self.root, "synth.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(SYNTHESISER)
        self.log = os.path.join(self.root, "voices.log")
        self.mode = os.path.join(self.root, "mode")
        self.settings = os.path.join(self.root, "settings.conf")
        with open(self.settings, "w", encoding="utf-8") as handle:
            handle.write(f"{settings.KEY_TTS_ENGINE}=mbrola\n")
        self.cfg = {"tts": {"cmd": [sys.executable, "-I", script, "{voice}", self.log, "ok"]}}
        patcher = mock.patch.dict(os.environ, {"XDG_DATA_DIRS": self.share,
                                               "GPU_TERMINAL_SETTINGS_FILE": self.settings})
        patcher.start()
        self.addCleanup(patcher.stop)

    def install(self, *databases: str, layout: str = "nested") -> None:
        for name in databases:
            if layout == "nested":
                path = os.path.join(self.share, "mbrola", name, name)
            elif layout == "flat":
                path = os.path.join(self.share, "mbrola", name)
            else:
                path = os.path.join(self.share, "mbrola", "voices", name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").close()

    def failing_mbrola(self) -> None:
        self.cfg["tts"]["cmd"][-1] = "mb-fails"

    def voices_run(self) -> list[str]:
        try:
            with open(self.log, encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        except FileNotFoundError:
            return []


class ResolutionTestCase(_MbrolaFixture):

    def test_a_language_resolves_to_its_most_preferred_installed_voice(self) -> None:
        cases = ((("us1", "us2"), "mb-us1"), (("us2", "us3"), "mb-us2"),
                 (("us3",), "mb-us3"))
        for installed, expected in cases:
            with self.subTest(installed=installed):
                self.setUp()
                self.install(*installed)
                self.assertEqual(tts.resolve_mbrola_voice("en-us"), expected)

    def test_every_layout_espeak_searches_counts_and_an_empty_directory_does_not(self) -> None:
        for layout in ("nested", "flat", "voices"):
            with self.subTest(layout=layout):
                self.setUp()
                self.install("us2", layout=layout)
                self.assertEqual(tts.resolve_mbrola_voice("en-us"), "mb-us2")
        self.setUp()
        os.makedirs(os.path.join(self.share, "mbrola", "us1"))     # no database in it
        with self.assertRaises(tts.TtsError):
            tts.resolve_mbrola_voice("en-us")

    def test_a_voice_built_on_another_languages_database_resolves_by_that_database(self) -> None:
        # Wave-3a survivor B2: de1-en speaks English with the de1 diphone
        # database, so de1 is the file that must exist, not one named de1-en.
        self.install("de1")
        self.assertEqual(tts.resolve_mbrola_voice("en"), "mb-de1-en")

    def test_an_explicit_voice_id_is_used_as_named(self) -> None:
        for voice, expected in (("us1", "mb-us1"), ("de4-en", "mb-de4-en"), ("US2", "mb-us2")):
            with self.subTest(voice=voice):
                self.assertEqual(tts.resolve_mbrola_voice(voice), expected)

    def test_a_language_with_nothing_installed_is_unavailable_naming_what_is(self) -> None:
        self.install("de4", "fr1")
        with self.assertRaises(tts.TtsError) as caught:
            tts.resolve_mbrola_voice("en-us")
        self.assertNotIsInstance(caught.exception, tts.TtsUnsupported)
        self.assertEqual(voiced._carried_code(caught.exception, "none"),
                         protocol.ERR_UNAVAILABLE)
        message = str(caught.exception)
        # What is installed: a database serves every voice built on it, so the
        # de4 and fr1 databases also install the de4-en and fr1-en voices.
        self.assertIn("installed MBROLA voices are: de4, de4-en, fr1, fr1-en.", message)
        self.assertIn("us1, us2, us3", message)                 # what speaks en-us
        self.assertIn("mbrola-us1", message)                    # what to install

    def test_a_language_no_mbrola_voice_speaks_is_unsupported(self) -> None:
        self.install("us1")
        with self.assertRaises(tts.TtsUnsupported) as caught:
            tts.resolve_mbrola_voice("tlh")
        self.assertEqual(caught.exception.code, protocol.ERR_UNSUPPORTED)
        self.assertIn("us1", str(caught.exception))


class EngineTestCase(_MbrolaFixture):

    def test_model_mbrola_with_the_default_voice_runs_the_installed_voice(self) -> None:
        self.install("us2")
        engine = tts.make_tts(self.cfg, model="mbrola")
        self.assertEqual((engine.voice, engine.selected_voice), ("en-us", "us2"))
        engine.synth("hello")
        self.assertEqual(self.voices_run(), ["mb-us2"])
        self.assertEqual((engine.last_provenance.model, engine.last_provenance.voice),
                         ("mbrola", "mb-us2"))

    def test_model_mbrola_with_nothing_installed_is_refused_before_anything_runs(self) -> None:
        self.install("de4")
        with self.assertRaises(tts.TtsError) as caught:
            tts.make_tts(self.cfg, model="mbrola")
        self.assertIn("de4", str(caught.exception))
        self.assertEqual(self.voices_run(), [])

    def test_the_settings_tier_with_nothing_installed_speaks_espeak_at_once(self) -> None:
        engine = tts.make_tts(self.cfg)                        # the settings select mbrola
        self.assertIsInstance(engine, tts.EspeakTts)
        self.assertEqual(engine.selected_voice, "en-us")
        engine.synth("hello")
        self.assertEqual(self.voices_run(), ["en-us"])          # no doomed mb- process
        self.assertEqual(engine.last_provenance.model, "espeak")
        self.assertIn("installed MBROLA voices are: none", engine.mbrola_error)

    def test_a_rendered_file_is_labelled_with_what_rendered_it(self) -> None:
        self.install("us1")
        rendered = tts.render_text("One. Two.", cfg=self.cfg)
        self.assertEqual((rendered.model, rendered.voice), ("mbrola", "us1"))
        self.failing_mbrola()
        rendered = tts.render_text("One. Two.", cfg=self.cfg)
        self.assertEqual((rendered.model, rendered.voice), ("espeak", "en-us"))

    def test_status_names_the_voice_the_tier_runs(self) -> None:
        self.install("us3")
        self.assertIn("mb-us3", tts.mbrola_selection_detail("en-us"))
        self.assertNotIn("mb-en-us", tts.mbrola_selection_detail("en-us"))
        self.setUp()
        detail = tts.mbrola_selection_detail("en-us")
        self.assertIn("plain espeak", detail)
        self.assertIn("none", detail)


class DaemonTestCase(_MbrolaFixture):
    """The speak request a client sends, through the daemon's dispatch."""

    def daemon(self):
        d = object.__new__(voiced.Daemon)
        d._session_dir = self.root
        d._cfg = self.cfg
        d._lock = threading.RLock()
        d._refresh_config = lambda: None
        d._touch = lambda: None
        d._warn = d._debug = lambda *a, **k: None
        d._next_turn_id = lambda kind: f"{kind}-1"
        d._speech = d._dictation = d._player = None
        d._speech_error = d._speech_error_turn = ""
        d._jobs = None
        d._arbiter = mock.Mock(listening=False, speaking=False)
        d._cancel_speech = lambda: None
        self.started = []
        d._start = lambda thread, undo: self.started.append(thread.name)
        return d

    def speak(self, **fields) -> dict:
        return voiced.Daemon._dispatch(self.daemon(), protocol.encode(
            dict(op="speak", text="Hello there.", **fields)))

    def test_speak_model_mbrola_is_accepted_naming_the_voice_that_runs(self) -> None:
        self.install("us1")
        reply = self.speak(model="mbrola")
        self.assertIs(reply["ok"], True, reply)
        self.assertEqual((reply["model"], reply["voice"]), ("mbrola", "us1"))
        self.assertEqual(len(self.started), 1)

    def test_speak_model_mbrola_with_nothing_installed_is_refused_with_its_code(self) -> None:
        self.install("fr1")
        reply = self.speak(model="mbrola")
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)
        self.assertIn("fr1", reply["error"])
        self.assertEqual(self.started, [])

    def test_speak_model_mbrola_in_a_language_it_lacks_is_unsupported(self) -> None:
        self.install("us1")
        reply = self.speak(model="mbrola", voice="tlh")
        self.assertEqual(reply.get("code"), protocol.ERR_UNSUPPORTED, reply)
        self.assertEqual(self.started, [])


if __name__ == "__main__":
    unittest.main()
