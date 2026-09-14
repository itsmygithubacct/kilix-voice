"""PIP-01 / A14: no descriptor claims a determinism its clip does not have.

Piper renders identical text with identical settings to different audio:
kilix-piper-tts sets only the length scale, and the model's noise is drawn
fresh on every synthesis. Every Piper descriptor still carried "seed": 0 as
though a seed had produced the clip. A seed cannot be given to the provider
from here, so the descriptor stops claiming one. It carries no seed, and its
settings say the output is not reproducible.

The provider is a hermetic fake whose output differs on every call, as the
real provider's does, reached through the trusted KILIX_PIPER_TTS override.
Clips go through the daemon's own path: _synth, then _play_if_current, then
_publish_chunk, which builds the descriptor and seals the WAV it describes.
espeak is the control. It is deterministic, and its descriptors keep an
integer seed and say so.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_determinism", importlib.machinery.SourceFileLoader(
        "kilix_voiced_determinism", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol, tts as tts_lib  # noqa: E402

# Like the real provider, each synthesis draws fresh noise: the same text gives
# different samples, and a slightly different length.
NOISY_PIPER = f"""#!{sys.executable}
import os, sys
sys.stdin.buffer.read()
frames = 800 + os.urandom(1)[0]
sys.stdout.buffer.write(os.urandom(frames * 2))
"""

STEADY_ESPEAK = r'''
import struct, sys
sys.stdin.buffer.read()
pcm = b"\x01\x00\x02\x00" * 400
fmt = struct.pack("<HHIIHH", 1, 1, 22050, 44100, 2, 16)
body = (b"fmt " + struct.pack("<I", 16) + fmt + b"data"
        + struct.pack("<I", len(pcm)) + pcm)
sys.stdout.buffer.write(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
'''

TEXT = "The same sentence, with the same settings."


class _PublishedClips(unittest.TestCase):

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = directory.name

    def publish(self, engine, turn_id: str) -> tuple[dict, bytes]:
        """One clip through the daemon's synth-and-publish path.

        Returns the descriptor as sent and the bytes of the WAV sent with it.
        """
        sent = []

        def send(receiver, message):
            fds = getattr(message, "fds", ())
            wav = os.pread(fds[0], 1 << 22, 0) if fds else b""
            sent.append((dict(message), wav))
            return True

        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._warn = lambda *a, **k: None
        d._send = send
        turn = voiced._SpeechTurn(turn_id, [TEXT], engine)
        turn.receiver = mock.Mock()
        d._speech = turn
        clip = voiced.Daemon._synth(d, turn, TEXT)
        pcm, rate = clip
        self.assertTrue(voiced.Daemon._play_if_current(
            d, turn, mock.Mock(), pcm, rate, clip.provenance))
        self.assertEqual(len(sent), 1, sent)
        descriptor, wav = sent[0]
        self.assertEqual(descriptor["sha256"], hashlib.sha256(wav).hexdigest())
        return descriptor, wav

    def piper(self) -> tts_lib.PiperTts:
        provider = os.path.join(self.dir, "kilix-piper-tts")
        with open(provider, "w", encoding="utf-8") as handle:
            handle.write(NOISY_PIPER)
        os.chmod(provider, 0o755)
        patcher = mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider})
        patcher.start()
        self.addCleanup(patcher.stop)
        return tts_lib.PiperTts(rate=170)

    def espeak(self) -> tts_lib.EspeakTts:
        script = os.path.join(self.dir, "espeak.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(STEADY_ESPEAK)
        return tts_lib.EspeakTts({"tts": {"cmd": [sys.executable, "-I", script]}},
                                 voice="en-us", rate=170)


class PiperDescriptors(_PublishedClips):

    def test_identical_requests_differ_and_no_descriptor_claims_a_seed(self) -> None:
        first, first_wav = self.publish(self.piper(), "speak-1")
        second, second_wav = self.publish(self.piper(), "speak-2")
        # The premise, observed rather than assumed: same text, same settings,
        # different audio.
        self.assertEqual(first["settings"]["rate_wpm"], second["settings"]["rate_wpm"])
        self.assertNotEqual(first_wav, second_wav)
        self.assertNotEqual(first["sha256"], second["sha256"])
        for descriptor in (first, second):
            with self.subTest(turn=descriptor["settings"]["turn"]):
                self.assertNotIn("seed", descriptor)
                self.assertEqual((descriptor["settings"]["reproducible"],
                                  descriptor["settings"]["seed_consumed"]),
                                 (False, False))
                self.assertEqual(descriptor["model"], "piper-en-us-kristin-medium")


class EspeakDescriptors(_PublishedClips):                          # control

    def test_identical_requests_are_identical_and_say_so(self) -> None:
        first, first_wav = self.publish(self.espeak(), "speak-1")
        second, second_wav = self.publish(self.espeak(), "speak-2")
        self.assertEqual(first_wav, second_wav)
        for descriptor in (first, second):
            with self.subTest(turn=descriptor["settings"]["turn"]):
                self.assertIsInstance(descriptor["seed"], int)          # V24 A14
                self.assertEqual(descriptor["seed"], 0)
                self.assertIs(descriptor["settings"]["reproducible"], True)


class ReportedSeeds(_PublishedClips):
    """A seed reaches a descriptor only when an engine reported one."""

    def engine(self, **attributes):
        engine = mock.Mock(model="thirdparty", voice="v1", rate=150,
                           effective_model=None, **attributes)
        engine.synth.side_effect = lambda text, **kw: (b"\x00\x00" * 64, 22050)
        return engine

    def test_an_engine_that_reports_a_seed_keeps_it(self) -> None:
        engine = self.engine(seed=42)
        del engine.last_provenance
        descriptor, _wav = self.publish(engine, "speak-3")
        self.assertEqual(descriptor["seed"], 42)
        self.assertIs(descriptor["settings"]["reproducible"], False)

    def test_an_engine_that_reports_none_is_given_none(self) -> None:
        for seed in (None, True, "7"):
            with self.subTest(seed=seed):
                engine = self.engine(seed=seed)
                del engine.last_provenance
                descriptor, _wav = self.publish(engine, "speak-4")
                self.assertNotIn("seed", descriptor)

    def test_a_recorded_provenance_without_a_seed_carries_none(self) -> None:
        engine = self.engine()
        engine.last_provenance = tts_lib.SynthesisProvenance("thirdparty", "v1",
                                                             rate_wpm=150)
        descriptor, _wav = self.publish(engine, "speak-5")
        self.assertNotIn("seed", descriptor)

    def test_the_constructor_adds_no_seed_it_was_not_given(self) -> None:
        chunk = protocol.synthesis_chunk(0, pcm_bytes=4, sample_rate=22050,
                                         voice="v1", model="m", reproducible=False)
        self.assertNotIn("seed", chunk)


if __name__ == "__main__":
    unittest.main()
