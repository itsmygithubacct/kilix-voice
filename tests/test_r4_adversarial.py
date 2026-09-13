"""R4 adversarial probes. Each test FAILS against the frozen subject and would
PASS against an implementation that closed the gap it names. Run with:
    /usr/bin/python3 -B -m unittest tests.test_r4_adversarial -v
"""
from __future__ import annotations

import importlib.util
import os
import threading
import time
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced", importlib.machinery.SourceFileLoader(
        "kilix_voiced", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol, tts as tts_lib  # noqa: E402


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class P2RecordingDeadlineTestCase(unittest.TestCase):
    """Property 2: the caller budget must bound the RECORDING loop, not only
    preparation. The turn carries a Cancellation with a spent budget; the
    microphone is delivering frames; a deadline that is 'honoured, not merely
    accepted' must stop feeding audio into the recogniser."""

    def test_recording_stops_when_the_caller_budget_is_spent(self) -> None:
        clock = _Clock()
        turn = voiced._DictationTurn("d-rec", mock.Mock(),
                                     deadline=clock() + 0.001, clock=clock)
        clock.advance(1.0)                       # budget already spent
        self.assertTrue(turn.stop.expired())
        self.assertFalse(turn.stop.is_set())     # NOT an explicit stop

        frames = [b"\x00" * 320] * 5
        capture = mock.Mock()
        capture.frame_bytes = 320
        capture.overruns = 0
        capture.error = ""
        capture.read.side_effect = lambda *a, **k: frames.pop(0) if frames else None

        engine = mock.Mock()
        engine.supports_partials = False
        engine.feed.return_value = None

        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"max_seconds": 120}, "vad": {"silence_ms": 1}}
        d._stopping = threading.Event()
        d._warn = lambda *a, **k: None
        d._debug = lambda *a, **k: None
        d._send = lambda *a, **k: True
        with mock.patch.object(voiced, "Vad",
                               lambda cfg: types.SimpleNamespace(feed=lambda f: "")):
            voiced.Daemon._record(d, turn, capture, engine)

        self.assertLessEqual(
            engine.feed.call_count, 1,
            "the recording loop fed %d frames into the recogniser after the "
            "caller deadline had already elapsed: deadline_ms is accepted for "
            "dictate but the recording loop is bounded only by max_seconds / "
            "VAD / explicit stop, never by turn.stop.expired()."
            % engine.feed.call_count)


class P2PiperPrepDeadlineTestCase(unittest.TestCase):
    """Property 2: Piper's availability probe (up to PIPER_STATUS_TIMEOUT_S =
    5 s) runs during preparation. _op_speak computes the budget first but calls
    engine.check_available() with no budget, so a sub-second deadline does not
    bound it."""

    def test_op_speak_bounds_the_piper_probe_by_the_caller_budget(self) -> None:
        engine = tts_lib.PiperTts(rate=170)      # real type: isinstance() holds
        engine.check_available = mock.Mock()

        d = object.__new__(voiced.Daemon)
        d._cfg = {}
        d._refresh_config = lambda: None
        d._speech_chunks = lambda text: []       # short-circuit after the probe

        with mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: engine):
            voiced.Daemon._op_speak(
                d, {"id": "r", "text": "hi", "deadline_ms": 50})

        engine.check_available.assert_called_once()
        budget = engine.check_available.call_args.kwargs.get("budget")
        self.assertIsNotNone(
            budget,
            "engine.check_available() was called with no budget, so Piper's "
            "5 s status probe can outlive a 50 ms request deadline.")
        self.assertLessEqual(budget, 0.06)


class P5ConsentDenialCodeTestCase(unittest.TestCase):
    """Property 5: a consent refusal must not be flattened to the same
    machine-readable code as a missing device. ERR_DENIED exists in the closed
    vocabulary and is never used; consent denial is reported as ERR_UNAVAILABLE,
    indistinguishable from 'no microphone'."""

    def test_a_consent_denial_carries_a_distinct_code(self) -> None:
        sent = []
        d = object.__new__(voiced.Daemon)
        d._warn = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda turn: None
        d._touch = lambda: None
        # The exact refusal _require_capture_consent raises in production.
        # Changed from DaemonError to ConsentDenied by the builder when fixing
        # THIS finding: the fix is a distinct exception class carrying
        # ERR_DENIED, so the class the reviewer modelled is exactly what
        # changed. The end-to-end test below drives the real
        # _require_capture_consent so this proxy cannot drift again.
        d._dictate = mock.Mock(side_effect=voiced.ConsentDenied(
            "dictation has no recorded consent for this configuration "
            "(model 'small-en-us', engine 'vosk'). The microphone was not "
            "opened."))
        turn = voiced._DictationTurn("d-consent", mock.Mock())
        voiced.Daemon._run_dictation(d, turn)

        self.assertEqual(len(sent), 1, sent)
        self.assertNotEqual(
            sent[0].get("code"), protocol.ERR_UNAVAILABLE,
            "a consent denial is reported with ERR_UNAVAILABLE, the same code "
            "as a missing microphone; the purpose-built ERR_DENIED is never "
            "used. A caller cannot tell 'you have not consented' from 'no "
            "device'.")
        self.assertEqual(sent[0].get("code"), protocol.ERR_DENIED)

    def test_the_real_consent_gate_produces_that_code_end_to_end(self) -> None:
        # Not a proxy: the refusal comes from _require_capture_consent itself,
        # so the mapping cannot drift away from what production raises.
        sent = []
        d = object.__new__(voiced.Daemon)
        d._warn = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda turn: None
        d._touch = lambda: None
        d._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent-model"}}
        turn = voiced._DictationTurn("d-e2e", mock.Mock())
        with mock.patch.dict(os.environ,
                             {"KILIX_VOICE_REQUIRE_CONSENT": "1"}), \
             mock.patch.object(voiced.consent, "granted",
                               lambda subject, digest: False):
            voiced.Daemon._run_dictation(d, turn)
        self.assertEqual(len(sent), 1, sent)
        self.assertEqual(sent[0]["code"], protocol.ERR_DENIED)
        self.assertIn("consent", sent[0].get("error", "").lower())


if __name__ == "__main__":
    unittest.main()
