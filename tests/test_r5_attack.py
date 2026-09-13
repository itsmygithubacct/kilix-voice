"""R5: independent verification of the three R4 fixes + fresh attacks on the
new code (a36d47f). VERIFY_* tests should PASS on the fixed tree. ATTACK_*
tests FAIL on the fixed tree and name a live defect."""
from __future__ import annotations

import importlib.util
import os
import threading
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


# --------------------------------------------------------------------------
# VERIFY: the three R4 findings, driven through the real production paths.
# --------------------------------------------------------------------------

class VerifyFinding3ConsentDenied(unittest.TestCase):
    def test_the_real_gate_raises_ConsentDenied_carrying_ERR_DENIED(self) -> None:
        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent"}}
        with mock.patch.dict(os.environ, {"KILIX_VOICE_REQUIRE_CONSENT": "1"}), \
             mock.patch.object(voiced.consent, "granted",
                               lambda subject, digest: False):
            with self.assertRaises(voiced.ConsentDenied) as caught:
                voiced.Daemon._require_capture_consent(
                    d, voiced.stt_lib.resolve_stt(d._cfg))
        self.assertEqual(caught.exception.code, protocol.ERR_DENIED)

    def test_the_worker_adapter_maps_a_carried_code(self) -> None:
        sent = []
        d = object.__new__(voiced.Daemon)
        d._warn = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda turn: None
        d._touch = lambda: None
        d._dictate = mock.Mock(side_effect=voiced.ConsentDenied("no consent"))
        voiced.Daemon._run_dictation(d, voiced._DictationTurn("d", mock.Mock()))
        self.assertEqual(sent[0].get("code"), protocol.ERR_DENIED)


class VerifyFinding1MidRecordingExpiry(unittest.TestCase):
    """A budget that expires mid-recording must abandon the turn: no `final`,
    and the caller learns it was the deadline."""

    def test_expiry_mid_recording_delivers_no_final_and_reports_deadline(self):
        clock = _Clock()
        turn = voiced._DictationTurn("d", mock.Mock(),
                                     deadline=clock() + 0.001, clock=clock)
        clock.advance(1.0)                       # spent, not explicitly stopped
        sent = []
        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent"},
                  "vad": {"silence_ms": 1}}
        d._stopping = threading.Event()
        d._warn = d._debug = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda t: None
        d._touch = lambda: None
        d._require_capture_consent = lambda resolved=None: None
        frames = [b"\x00" * 320] * 5
        capture = mock.Mock()
        capture.rate = 16000
        capture.frame_bytes = 320
        capture.overruns = 0
        capture.error = ""
        capture.read.side_effect = lambda *a, **k: frames.pop(0) if frames else None
        engine = mock.Mock()
        engine.supports_partials = False
        engine.feed.return_value = None
        engine.end_utterance.return_value = "words that must not be delivered"
        with mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture), \
             mock.patch.object(voiced.stt_lib, "make_stt",
                               lambda *a, **k: engine), \
             mock.patch.object(voiced, "Vad",
                               lambda cfg: types.SimpleNamespace(feed=lambda f: "")):
            voiced.Daemon._run_dictation(d, turn)
        self.assertEqual(len(sent), 1, sent)
        self.assertNotIn("final", sent[0], "delivered a transcript after expiry")
        self.assertEqual(sent[0].get("code"), protocol.ERR_DEADLINE)


# --------------------------------------------------------------------------
# ATTACK: fresh defects in the new code.
# --------------------------------------------------------------------------

class AttackPiperDeadlineMiscoded(unittest.TestCase):
    """The finding-2 fix bounds the Piper probe by the caller budget. When the
    probe is cut short BY that budget, piper_status returns a message that says
    'the request deadline elapsed', but check_available raises TtsError and the
    dispatch maps every TtsError to ERR_INTERNAL. A caller that set too tight a
    deadline is told the daemon has a bug, not that its deadline was too short."""

    def test_a_speak_deadline_spent_in_the_piper_probe_reports_deadline(self):
        engine = tts_lib.PiperTts(rate=170)
        # Exactly what piper_status(budget<=0) yields, surfaced by check_available.
        engine.check_available = mock.Mock(side_effect=tts_lib.TtsError(
            "the request deadline elapsed before kilix-piper-tts could be "
            "inspected"))
        d = object.__new__(voiced.Daemon)
        d._session_dir = "/tmp"
        d._cfg = {}
        d._refresh_config = lambda: None
        d._touch = lambda: None
        with mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: engine):
            reply = voiced.Daemon._dispatch(
                d, protocol.encode({"op": "speak", "text": "hi",
                                    "deadline_ms": 1}))
        self.assertFalse(reply["ok"])
        self.assertEqual(
            reply["code"], protocol.ERR_DEADLINE,
            "a deadline spent during the Piper probe is reported as %r, not "
            "'deadline'; the same class of defect as R4 finding 3."
            % reply["code"])


class AttackBroadArmForwardsInvalidCode(unittest.TestCase):
    """The worker adapter does `getattr(error, "code", None) or ERR_UNAVAILABLE`
    and passes the result straight to dictation_error, which REJECTS a code
    outside the closed vocabulary. An exception whose .code is not a valid code
    makes dictation_error raise, so the caller gets NO datagram at all -- worse
    than the flatten it replaced. Not reachable by today's production
    exceptions (only ConsentDenied carries .code), so this is latent, but the
    adapter trusts an attribute it should validate."""

    def test_an_unvalidated_carried_code_still_yields_one_valid_datagram(self):
        class WeirdError(voiced.DaemonError):
            code = "not-a-real-code"
        sent = []
        d = object.__new__(voiced.Daemon)
        d._warn = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda t: None
        d._touch = lambda: None
        d._dictate = mock.Mock(side_effect=WeirdError("boom"))
        # A robust adapter delivers exactly one datagram with a valid code.
        voiced.Daemon._run_dictation(d, voiced._DictationTurn("d", mock.Mock()))
        self.assertEqual(len(sent), 1,
                         "the caller received %d datagrams, not 1" % len(sent))
        self.assertIn(sent[0].get("code"), protocol.ERROR_CODES)


if __name__ == "__main__":
    unittest.main()
