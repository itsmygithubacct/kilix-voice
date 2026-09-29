"""Whisper dictation through a persistent provider child, bound to consent.

A stand-in ``kilix-whisper-stt`` speaks the provider's serve protocol. It
reads every model file through the directory it is given when it starts, as
the real provider loads its model, and logs what it read and each request,
so the tests can see which bytes loaded and how many decodes ran. Its reply
is empty for silence and names the bytes it loaded otherwise.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from tests.test_dictation_waits_for_words import FrameCapture, pop_then_words
from tests.test_terminal_outcomes import _LiveTurnsFixture, voiced
from voicelib import consent, models, paths, stt

FAKE_PROVIDER = r"""#!{python}
import hashlib, json, os, sys, time
args = sys.argv[1:]
log = open(os.environ["FAKE_WHISPER_LOG"], "a")
model = args[args.index("--model") + 1]
loaded = {{name: hashlib.sha256(open(os.path.join(model, name), "rb").read()).hexdigest()
          for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")}}
log.write(json.dumps({{"start": os.getpid(), "loaded": loaded, "argv": args}}) + "\n"); log.flush()
mode = os.environ.get("FAKE_WHISPER_MODE", "")
if mode == "load-error":
    print(json.dumps({{"error": "cannot load the model"}}), flush=True); sys.exit(1)
if mode == "crash":
    sys.exit(7)
if mode == "slow-load":
    time.sleep(1)
print(json.dumps({{"ready": True, "version": "0.1.0"}}), flush=True)
stdin = sys.stdin.buffer
while True:
    header = stdin.readline()
    if not header:
        sys.exit(0)
    size = json.loads(header)["pcm_bytes"]
    pcm = stdin.read(size)
    log.write(json.dumps({{"request": len(pcm)}}) + "\n"); log.flush()
    if mode == "hang":
        time.sleep(3600)
    if mode == "error":
        print(json.dumps({{"error": "decoder exploded"}}), flush=True); continue
    text = "" if not pcm.strip(b"\0") else "words from " + loaded["model.bin"][:8]
    print(json.dumps({{"text": text}}), flush=True)
"""

FILES = {"model.bin": b"weights A", "config.json": b"{}", "tokenizer.json": b"{}",
         "vocabulary.txt": b"a\nb\n"}
SPEECH = b"\x01\x10" * 320
SILENCE = b"\x00\x00" * 320


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


class WhisperFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="kv-whisper-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.log = self.root / "provider.log"
        provider = self.root / "kilix-whisper-stt"
        provider.write_text(FAKE_PROVIDER.format(python=sys.executable))
        provider.chmod(0o755)
        env = {"KILIX_DATA_HOME": str(self.root / "data"),
               "KILIX_SESSION_HOME": str(self.root / "session"),
               stt.ENV_WHISPER: str(provider), "FAKE_WHISPER_LOG": str(self.log)}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(stt.ENV_MODEL, None)
        self.model = Path(paths.model_dir(models.WHISPER_MODEL))
        self.model.mkdir(parents=True)
        for name, data in FILES.items():
            (self.model / name).write_bytes(data)

    def entries(self, key):
        if not self.log.exists():
            return []
        return [row for row in map(json.loads, self.log.read_text().splitlines()) if key in row]

    def engine(self, **options):
        engine = stt.WhisperStt(16000, model_path=str(self.model), **options)
        self.addCleanup(engine.close)
        return engine

    def turn(self, engine, *frames):
        engine.start_utterance()
        for frame in frames:
            engine.feed(frame)
        return engine.end_utterance()


class WhisperEngineTests(WhisperFixture):
    def test_the_model_lives_in_the_content_asset_directory(self):
        self.assertEqual(self.model, self.root / "data/desktop-apps/assets/faster-whisper-small-en")
        local = self.root / "data/voice/models" / models.WHISPER_MODEL
        local.mkdir(parents=True)
        self.assertEqual(paths.model_dir(models.WHISPER_MODEL), str(local))   # a local copy wins
        local.rmdir()
        resolved = stt.resolve_stt({"stt": {"engine": "whisper", "model": "small-en-us"}})
        self.assertEqual((resolved.engine, resolved.model_id, resolved.model_dir),
                         ("whisper", models.WHISPER_MODEL, str(self.model)))
        engine = stt.make_stt({"stt": {"engine": "whisper"}}, 16000, resolved=resolved)
        self.addCleanup(engine.close)
        self.assertIsInstance(engine, stt.WhisperStt)

    def test_one_child_serves_every_turn(self):
        engine = self.engine(threads=3)
        self.assertEqual(self.turn(engine, SPEECH, SPEECH), "words from " + sha(b"weights A")[:8])
        self.assertEqual(self.turn(engine, SILENCE), "")
        self.assertEqual(self.turn(engine), "")                   # no audio, no request
        self.assertEqual(len(self.entries("start")), 1)
        self.assertEqual([row["request"] for row in self.entries("request")],
                         [len(SPEECH) * 2, len(SILENCE)])
        argv = self.entries("start")[0]["argv"]
        self.assertEqual(argv[0], "serve")
        self.assertEqual(argv[argv.index("--threads") + 1], "3")

    def test_the_child_loads_the_held_bytes_not_the_path(self):
        payload = consent.payload_digest_at(str(self.model), "whisper")
        engine = self.engine(consented_payload=payload)
        # Replaced after the gate: the path now names other bytes. The child
        # may even start loading after this; it still reads the held files.
        swap = self.model / "model.bin.new"
        swap.write_bytes(b"weights B")
        os.replace(swap, self.model / "model.bin")
        self.assertEqual(self.turn(engine, SPEECH), "words from " + sha(b"weights A")[:8])
        self.assertEqual(self.entries("start")[0]["loaded"]["model.bin"], sha(b"weights A"))

    def test_a_payload_other_than_the_consented_one_is_never_started(self):
        with self.assertRaisesRegex(stt.SttError, "changed after dictation consent"):
            stt.WhisperStt(16000, model_path=str(self.model), consented_payload="0" * 64)
        time.sleep(0.2)
        self.assertEqual(self.entries("start"), [])

    def test_a_held_file_rewritten_in_place_discards_the_transcript(self):
        engine = self.engine(consented_payload=consent.payload_digest_at(str(self.model), "whisper"))
        self.assertTrue(self.turn(engine, SPEECH))                # loaded and answering
        with open(self.model / "vocabulary.txt", "r+b") as handle:
            handle.write(b"x")                                    # same inode, new bytes
        with self.assertRaisesRegex(stt.SttError, "modified while it was being transcribed"):
            self.turn(engine, SPEECH)

    def test_bytes_changed_while_the_model_loads_never_hear_the_audio(self):
        os.environ["FAKE_WHISPER_MODE"] = "slow-load"
        engine = self.engine(consented_payload=consent.payload_digest_at(str(self.model), "whisper"))
        with open(self.model / "vocabulary.txt", "r+b") as handle:
            handle.write(b"x")                                    # during the load
        with self.assertRaisesRegex(stt.SttError, "modified while the model was loading"):
            self.turn(engine, SPEECH)
        self.assertEqual(self.entries("request"), [])

    def test_files_written_moments_ago_are_always_rehashed(self):
        # Equal stats prove nothing for a file written in the snapshot's own
        # timestamp tick: an equal-length rewrite leaves them all unchanged.
        engine = self.engine()
        with mock.patch.object(consent, "payload_digest_of",
                               wraps=consent.payload_digest_of) as digest:
            engine._require_unchanged("in the test")
            self.assertEqual(digest.call_count, 1)
            engine._stats_taken += 10 * 1_000_000_000         # long after the writes
            engine._require_unchanged("in the test")
            self.assertEqual(digest.call_count, 1)

    def test_close_kills_the_child_and_removes_its_links(self):
        engine = self.engine()
        self.turn(engine, SPEECH)
        pid = self.entries("start")[0]["start"]
        links = engine._links
        self.assertTrue(alive(pid))
        engine.close()
        deadline = time.monotonic() + 5
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(alive(pid))
        self.assertFalse(os.path.exists(links))
        self.assertEqual(engine._held, {})
        with self.assertRaises(stt.SttError):
            engine.start_utterance()

    def test_close_from_another_thread_ends_a_decode_in_progress(self):
        os.environ["FAKE_WHISPER_MODE"] = "hang"
        engine = self.engine()
        engine.start_utterance()
        engine.feed(SPEECH)
        threading.Timer(0.5, engine.close).start()
        started = time.monotonic()
        self.assertEqual(engine.end_utterance(), "")
        self.assertLess(time.monotonic() - started, 10)

    def test_a_wedged_child_is_bounded_and_killed(self):
        os.environ["FAKE_WHISPER_MODE"] = "hang"
        engine = self.engine()
        with mock.patch.object(stt, "WHISPER_BASE_TIMEOUT_S", 0.5), \
                mock.patch.object(stt, "WHISPER_TIMEOUT_PER_AUDIO_S", 0.0):
            with self.assertRaisesRegex(stt.SttError, "did not answer"):
                self.turn(engine, SPEECH)
        pid = self.entries("start")[0]["start"]
        time.sleep(0.2)
        self.assertFalse(alive(pid))
        with self.assertRaisesRegex(stt.SttError, "stopped after an earlier failure"):
            self.turn(engine, SPEECH)

    def test_provider_failures_are_named(self):
        for mode, pattern in (("error", "decoder exploded"),
                              ("load-error", "cannot load the model"),
                              ("crash", "exited with status 7")):
            with self.subTest(mode=mode):
                os.environ["FAKE_WHISPER_MODE"] = mode
                engine = self.engine()
                with self.assertRaisesRegex(stt.SttError, pattern):
                    self.turn(engine, SPEECH)

    def test_a_missing_runtime_names_the_install_command_and_never_uses_path(self):
        os.environ.pop(stt.ENV_WHISPER)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        shutil.copy(self.root / "kilix-whisper-stt", bin_dir)    # on PATH, never used
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}):
            with self.assertRaisesRegex(stt.SttError, "kilix stt --install whisper-small-en"):
                stt.WhisperStt(16000, model_path=str(self.model))
        self.assertEqual(stt.whisper_binary(),
                         str(self.root / "data/voice/whisper/current/bin/kilix-whisper-stt"))

    def test_a_missing_model_file_is_named(self):
        (self.model / "tokenizer.json").unlink()
        with self.assertRaisesRegex(stt.SttError, "tokenizer.json is missing"):
            stt.WhisperStt(16000, model_path=str(self.model))


class ProvisionalTests(WhisperFixture):
    def test_a_wordless_segment_is_dropped_and_words_are_not_decoded_twice(self):
        engine = self.engine()
        engine.start_utterance()
        engine.feed(SILENCE)
        self.assertEqual(engine.provisional(), "")                # the pop: dropped
        engine.feed(SPEECH)
        words = engine.provisional()
        self.assertTrue(words)
        self.assertEqual(engine.end_utterance(), words)
        # Two decodes: the pop alone, then only the speech. None at the end.
        self.assertEqual([row["request"] for row in self.entries("request")],
                         [len(SILENCE), len(SPEECH)])

    def test_audio_after_a_provisional_is_decoded_again(self):
        engine = self.engine()
        engine.start_utterance()
        engine.feed(SPEECH)
        engine.provisional()
        engine.feed(SPEECH)
        engine.end_utterance()
        self.assertEqual([row["request"] for row in self.entries("request")],
                         [len(SPEECH), len(SPEECH) * 2])

    def test_a_new_turn_forgets_the_last_provisional(self):
        engine = self.engine()
        engine.start_utterance()
        engine.feed(SPEECH)
        engine.provisional()
        engine.start_utterance()
        engine.feed(SILENCE)
        self.assertEqual(engine.end_utterance(), "")


class WholeUtterancePopTests(_LiveTurnsFixture):
    """The daemon keeps listening through a pop for an end-of-turn engine."""

    def record(self, engine_class):
        frames, speech_from = pop_then_words()
        capture = FrameCapture(frames)
        engine = engine_class(speech_from)
        turn = SimpleNamespace(stop=voiced.Cancellation(), progress_lock=threading.Lock(),
                               receiver=None, hold=False, silence_ms=1500, owner="",
                               id="listen-test")
        self.daemon._send = lambda *args, **kwargs: True
        engine.start_utterance()
        self.daemon._record(turn, capture, engine)
        fed = engine.fed
        return engine, fed, speech_from, engine.end_utterance()

    def test_a_pop_then_speech_delivers_the_speech_decoded_once(self):
        engine, fed, speech_from, text = self.record(BufferedStandIn)
        self.assertEqual(text, "hello there")
        self.assertGreater(fed, speech_from + 25)                 # listened into the speech
        # Two decodes and none at the end: the pop's segment, dropped, then
        # only the audio after it.
        self.assertEqual(len(engine.decodes), 2)
        self.assertEqual(engine.decodes[1], fed * 640 - engine.decodes[0])

    def test_an_engine_that_did_not_opt_in_keeps_the_energy_rule(self):
        engine, fed, speech_from, _text = self.record(NotOptedIn)
        self.assertLess(fed, speech_from)


class BufferedStandIn(stt._WholeUtterance):
    """A whole-utterance engine whose decode hears words only in speech."""

    label = "stand-in"

    def __init__(self, speech_from):
        self.speech_from = speech_from
        self.fed = 0
        self.decodes = []
        self._closed = False
        self._open = False
        self._reset_turn()

    def feed(self, frame):
        self.fed += 1
        self.speech = self.fed > self.speech_from
        return super().feed(frame)

    def _decode(self, pcm):
        self.decodes.append(len(pcm))
        return "hello there" if self.speech else ""


class NotOptedIn(BufferedStandIn):
    transcribes_at_end = False


if __name__ == "__main__":
    unittest.main()
