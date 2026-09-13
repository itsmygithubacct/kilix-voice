"""The five open R3 findings, each as a test that fails against the old code.

F01 speech expiry did not stop the player or bound synthesis in progress
F02 dictation had no execution deadline and checked stop too late
F03 consent identity was not derived from the recogniser's configuration
F07 chunk-socket ownership was not unwound on dispatch refusal
F08 the capture refusal lost its machine-readable reason at the worker boundary

Every test here asserts an EFFECT -- a call made, a device not opened, a code
on the wire. None of them greps the source, because two tests in the previous
round did and broke against correct code while passing against a defect.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced", importlib.machinery.SourceFileLoader(
        "kilix_voiced", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import consent, protocol, stt as stt_lib, tts as tts_lib  # noqa: E402
from voicelib.cancel import Cancellation, Cancelled, DeadlineExceeded  # noqa: E402


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _speech_daemon(turn, player):
    d = object.__new__(voiced.Daemon)
    d._lock = threading.RLock()
    d._speech = turn
    d._warn = lambda *a, **k: None
    d._touch = lambda: None
    d._arbiter = mock.Mock()
    d._report_speech_failure = lambda t, m: None
    d._get_player = lambda: player
    return d


def _engine():
    e = mock.Mock()
    e.voice, e.model, e.rate, e.seed = "en-us", "m1", 170, 7
    e.effective_model = None
    e.synth.side_effect = lambda text, **kw: (b"\x00\x00", 24000)
    return e


class F01SpeechExpiryTestCase(unittest.TestCase):
    """Expiry must stop the PLAYER, and must bound the blocking engine call."""

    def test_an_expired_turn_stops_the_player(self) -> None:
        clock = _Clock()
        turn = voiced._SpeechTurn("s1", ["one"], _engine(), clock() + 5.0, clock)
        turn.receiver = None
        player = mock.Mock()
        player.error = ""
        # Still "playing" when the budget runs out: the Player is a separate
        # worker draining its own queue, which is exactly the case where
        # releasing ownership was not enough.
        player.playing = True
        # The budget dies WHILE the clip is playing. Driven through the
        # injected clock rather than by sleeping, so the test is deterministic
        # and cannot hang the way an earlier mutant made one hang.
        player.play.side_effect = lambda *a, **k: clock.advance(10.0)
        voiced.Daemon._run_speech(_speech_daemon(turn, player), turn)
        player.play.assert_called_once()          # it really did start playing
        player.stop.assert_called_once()          # and it really was stopped

    def test_a_replaced_turn_does_not_stop_its_successors_audio(self) -> None:
        # The identity guard. An old worker finishing late must not silence
        # the turn that replaced it.
        clock = _Clock()
        turn = voiced._SpeechTurn("s-old", ["one"], _engine(), clock() + 0.001, clock)
        turn.receiver = None
        clock.advance(1.0)               # already expired
        player = mock.Mock()
        player.playing = False
        player.error = ""
        daemon = _speech_daemon(turn, player)
        daemon._speech = voiced._SpeechTurn("s-new", ["x"], _engine())  # replaced
        voiced.Daemon._run_speech(daemon, turn)
        player.stop.assert_not_called()

    def test_a_completed_turn_is_not_cut_off(self) -> None:      # control
        turn = voiced._SpeechTurn("s-ok", ["one"], _engine())    # no deadline
        turn.receiver = None
        player = mock.Mock()
        player.playing = False
        player.error = ""
        voiced.Daemon._run_speech(_speech_daemon(turn, player), turn)
        player.stop.assert_not_called()

    def test_the_engine_is_given_the_remaining_budget(self) -> None:
        clock = _Clock()
        turn = voiced._SpeechTurn("s2", ["one"], _engine(), clock() + 3.0, clock)
        daemon = object.__new__(voiced.Daemon)
        voiced.Daemon._synth(daemon, turn, "one")
        self.assertEqual(turn.engine.synth.call_args.kwargs["budget"], 3.0)

    def test_an_engine_ceiling_cannot_outlive_the_request(self) -> None:
        # 210 s of Piper inside a 2 s request must become 2 s.
        self.assertEqual(tts_lib._bounded(210.0, 2.0), 2.0)
        self.assertEqual(tts_lib._bounded(210.0, None), 210.0)
        with self.assertRaises(tts_lib.TtsError):
            tts_lib._bounded(210.0, 0.0)

    def test_piper_status_keeps_its_contract_on_a_spent_budget(self) -> None:
        ok, detail = tts_lib.piper_status(budget=0.0)
        self.assertFalse(ok)
        self.assertIn("deadline", detail)


class F02DictationDeadlineTestCase(unittest.TestCase):
    """Dictation must honour the budget it accepts, and stop before the mic."""

    def test_the_turn_carries_the_caller_budget(self) -> None:
        clock = _Clock()
        turn = voiced._DictationTurn("d1", mock.Mock(), clock() + 2.0, clock)
        self.assertFalse(turn.expired())
        clock.advance(3.0)
        self.assertTrue(turn.expired())
        self.assertEqual(turn.stop.reason(), "deadline")

    def test_a_stop_during_preparation_keeps_the_microphone_shut(self) -> None:
        opened = []
        daemon = object.__new__(voiced.Daemon)
        daemon._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent"}}
        daemon._require_capture_consent = lambda resolved=None: None
        turn = voiced._DictationTurn("d2", mock.Mock())
        turn.stop.set()                  # stop arrives BEFORE the worker runs
        with mock.patch.object(voiced.audio, "MicCapture",
                               lambda cfg: opened.append("mic")), \
             mock.patch.object(voiced.stt_lib, "make_stt",
                               lambda *a, **k: mock.Mock()):
            with self.assertRaises(Cancelled):
                voiced.Daemon._dictate(daemon, turn)
        self.assertEqual(opened, [])

    def test_an_expired_budget_keeps_the_microphone_shut(self) -> None:
        opened = []
        clock = _Clock()
        daemon = object.__new__(voiced.Daemon)
        daemon._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent"}}
        daemon._require_capture_consent = lambda resolved=None: None
        turn = voiced._DictationTurn("d3", mock.Mock(), clock() + 0.001, clock)
        clock.advance(1.0)
        with mock.patch.object(voiced.audio, "MicCapture",
                               lambda cfg: opened.append("mic")), \
             mock.patch.object(voiced.stt_lib, "make_stt",
                               lambda *a, **k: mock.Mock()):
            with self.assertRaises(DeadlineExceeded):
                voiced.Daemon._dictate(daemon, turn)
        self.assertEqual(opened, [])


class F02DispatchTestCase(unittest.TestCase):
    """The dispatch path must refuse a spent budget before claiming the mic."""

    def test_a_spent_budget_is_refused_at_dispatch(self) -> None:
        sock = mock.Mock()
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._cfg = {"stt": {"engine": "vosk"}}
        d._refresh_config = lambda: None
        d._next_turn_id = lambda kind: f"{kind}-1"
        d._arbiter = mock.Mock()
        d._arbiter.listening = False
        started = []
        d._start = lambda thread, undo: started.append("started")
        with mock.patch.object(voiced, "_connect_dictation", lambda p: sock):
            reply = voiced.Daemon._op_dictate(
                d, {"id": "r", "sock": "/tmp/s", "deadline_ms": 0})
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE)
        self.assertEqual(started, [])           # nothing claimed the mic
        sock.close.assert_called_once()         # and nothing was leaked

    def test_a_live_budget_still_dispatches(self) -> None:          # control
        sock = mock.Mock()
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._cfg = {"stt": {"engine": "vosk"}}
        d._refresh_config = lambda: None
        d._next_turn_id = lambda kind: f"{kind}-1"
        d._arbiter = mock.Mock()
        d._arbiter.listening = False
        d._start = lambda thread, undo: None
        d._clear_dictation = lambda turn: None
        with mock.patch.object(voiced, "_connect_dictation", lambda p: sock):
            reply = voiced.Daemon._op_dictate(
                d, {"id": "r", "sock": "/tmp/s", "deadline_ms": 60_000})
        self.assertTrue(reply["ok"])


class F03ConsentIdentityTestCase(unittest.TestCase):
    """Consent must describe the recogniser that is actually opened."""

    def test_one_resolution_feeds_both_consent_and_construction(self) -> None:
        cfg = {"stt": {"engine": "vosk", "model_path": "/tmp/model-A"}}
        resolved = stt_lib.resolve_stt(cfg)
        self.assertEqual(resolved.model_dir, "/tmp/model-A")
        seen = {}
        with mock.patch.object(stt_lib, "VoskStt",
                               lambda rate, **kw: seen.update(kw) or mock.Mock()):
            stt_lib.make_stt(cfg, 16000, resolved=resolved)
        # The recogniser opens the SAME directory the consent digest binds.
        self.assertEqual(seen["model_path"], resolved.model_dir)

    def test_the_model_path_override_moves_the_consent_identity(self) -> None:
        # The concrete F03 failure: consent hashed the CATALOGUE path while the
        # recogniser opened stt.model_path. Changing the override must change
        # what consent is bound to; if it does not, the two have drifted apart.
        a = stt_lib.resolve_stt({"stt": {"engine": "vosk",
                                         "model_path": "/tmp/model-A"}})
        b = stt_lib.resolve_stt({"stt": {"engine": "vosk",
                                         "model_path": "/tmp/model-B"}})
        self.assertNotEqual(a.model_dir, b.model_dir)

    def test_the_payload_digest_reads_the_resolved_directory(self) -> None:
        import tempfile
        from voicelib import models
        engine = next(iter(models.REQUIRED_FILES))
        required = models.REQUIRED_FILES[engine]
        with tempfile.TemporaryDirectory() as root:
            for rel in required:
                target = os.path.join(root, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(b"payload-one")
            first = consent.payload_digest_at(root, engine)
            self.assertTrue(first)
            # Same names, different bytes, SAME length: the case the removed
            # (size, mtime) cache used to get wrong.
            for rel in required:
                with open(os.path.join(root, rel), "wb") as fh:
                    fh.write(b"payload-two")
            self.assertNotEqual(first, consent.payload_digest_at(root, engine))

    def test_the_gate_hashes_the_directory_the_turn_resolved(self) -> None:
        # The daemon-level binding, not just the resolver's. A mutant that
        # re-derives the identity from settings inside the consent helper --
        # which is exactly what F03 described -- hashes the CATALOGUE path and
        # fails here, while every resolver-level test above still passes.
        seen = {}
        daemon = object.__new__(voiced.Daemon)
        daemon._cfg = {"stt": {"engine": "vosk", "model_path": "/tmp/model-X"}}
        resolved = stt_lib.resolve_stt(daemon._cfg)
        with mock.patch.dict(os.environ,
                             {"KILIX_VOICE_REQUIRE_CONSENT": "1"}), \
             mock.patch.object(consent, "payload_digest_at",
                               lambda root, engine: seen.update(root=root) or ""), \
             mock.patch.object(consent, "capture_digest",
                               lambda m, e, p: seen.update(model=m, engine=e) or "d"), \
             mock.patch.object(consent, "granted", lambda subject, digest: True):
            voiced.Daemon._require_capture_consent(daemon, resolved)
        self.assertEqual(seen["root"], "/tmp/model-X")
        self.assertEqual(seen["engine"], "vosk")

    def test_an_absent_directory_binds_nothing_rather_than_guessing(self) -> None:
        self.assertEqual(consent.payload_digest_at(None, "vosk"), "")
        self.assertEqual(consent.payload_digest_at("/nonexistent", "vosk"), "")


class F07ChunkSocketOwnershipTestCase(unittest.TestCase):
    """The chunk receiver must be released on every pre-worker failure."""

    def _daemon(self):
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._speech = None
        d._speech_error = ""
        d._arbiter = mock.Mock()
        d._cfg = {}
        d._refresh_config = lambda: None
        d._speech_chunks = lambda text: ["one"]
        d._next_turn_id = lambda kind: f"{kind}-1"
        d._cancel_speech = lambda: None
        d._warn = lambda *a, **k: None
        return d

    def _request(self, deadline_ms):
        return {"id": "r1", "text": "hello", "sock": "/tmp/x",
                "chunk_sock": "/tmp/chunks", "deadline_ms": deadline_ms}

    def test_an_expired_request_closes_the_chunk_socket(self) -> None:
        sock = mock.Mock()
        d = self._daemon()
        with mock.patch.object(voiced, "_connect_dictation", lambda p: sock), \
             mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: _engine()):
            reply = voiced.Daemon._op_speak(d, self._request(0))
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE)
        sock.close.assert_called_once()

    def test_an_arbiter_refusal_closes_the_chunk_socket(self) -> None:
        sock = mock.Mock()
        d = self._daemon()
        d._arbiter.begin_speech.side_effect = voiced.ArbiterError("busy")
        with mock.patch.object(voiced, "_connect_dictation", lambda p: sock), \
             mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: _engine()):
            with self.assertRaises(voiced.ArbiterError):
                voiced.Daemon._op_speak(d, self._request(None))
        sock.close.assert_called_once()

    def test_a_worker_that_will_not_start_closes_the_chunk_socket(self) -> None:
        sock = mock.Mock()
        d = self._daemon()
        d._start = mock.Mock(side_effect=voiced.DaemonError("no thread"))
        with mock.patch.object(voiced, "_connect_dictation", lambda p: sock), \
             mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: _engine()):
            with self.assertRaises(voiced.DaemonError):
                voiced.Daemon._op_speak(d, self._request(None))
        sock.close.assert_called_once()

    def test_a_started_worker_keeps_the_socket(self) -> None:     # control
        sock = mock.Mock()
        d = self._daemon()
        d._start = lambda thread, undo: None
        with mock.patch.object(voiced, "_connect_dictation", lambda p: sock), \
             mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: _engine()):
            reply = voiced.Daemon._op_speak(d, self._request(None))
        self.assertTrue(reply["ok"])
        # Ownership transferred: closing it here would break the stream.
        sock.close.assert_not_called()


class F08WorkerErrorCodeTestCase(unittest.TestCase):
    """Specific refusals must survive the worker adapter with their codes."""

    def _run_with(self, error):
        sent = []
        d = object.__new__(voiced.Daemon)
        d._warn = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda turn: None
        d._touch = lambda: None
        d._dictate = mock.Mock(side_effect=error)
        turn = voiced._DictationTurn("d-err", mock.Mock())
        voiced.Daemon._run_dictation(d, turn)
        self.assertEqual(len(sent), 1, sent)
        return sent[0]

    def test_an_oversized_capture_keeps_its_own_code(self) -> None:
        msg = self._run_with(protocol.MessageTooLarge("1 GB of audio"))
        self.assertEqual(msg["code"], protocol.ERR_TOO_LARGE)

    def test_a_deadline_is_not_reported_as_unavailable(self) -> None:
        msg = self._run_with(DeadlineExceeded("the request deadline elapsed"))
        self.assertEqual(msg["code"], protocol.ERR_DEADLINE)

    def test_a_stop_is_reported_as_cancelled(self) -> None:
        msg = self._run_with(Cancelled("the turn was stopped"))
        self.assertEqual(msg["code"], protocol.ERR_CANCELLED)

    def test_an_unexpected_failure_carries_a_code_at_all(self) -> None:
        msg = self._run_with(ZeroDivisionError("boom"))
        self.assertEqual(msg["code"], protocol.ERR_INTERNAL)

    def test_a_missing_device_still_reads_unavailable(self) -> None:   # control
        msg = self._run_with(voiced.DaemonError("no microphone"))
        self.assertEqual(msg["code"], protocol.ERR_UNAVAILABLE)


class R4SurvivorCoverageTestCase(unittest.TestCase):
    """The four mutations R4 wrote that survived all 540 tests.

    Each is correct code with nothing guarding it, which is the same class of
    hole as an untested fix: it can be deleted and no named check objects.
    """

    def test_an_explicit_cancel_outranks_expiry_in_the_reason(self) -> None:
        # R4 N1. A turn that was stopped and then sat past its deadline was
        # STOPPED; reporting "deadline" misattributes it. The docstring
        # insisted on this ordering and nothing enforced it.
        clock = _Clock()
        token = Cancellation(clock() + 0.001, clock)
        clock.advance(1.0)
        self.assertEqual(token.reason(), "deadline")
        token.set()
        self.assertEqual(token.reason(), "cancelled")

    def test_recording_stops_when_the_daemon_is_shutting_down(self) -> None:
        # R4 N3. Deleting the _stopping check left every test green.
        engine = mock.Mock()
        engine.supports_partials = False
        capture = mock.Mock()
        capture.frame_bytes = 320
        capture.overruns = 0
        capture.error = ""
        capture.read.side_effect = lambda *a, **k: b"\x00" * 320
        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"max_seconds": 120}, "vad": {"silence_ms": 1}}
        d._stopping = threading.Event()
        d._stopping.set()                     # daemon is going down
        d._warn = d._debug = lambda *a, **k: None
        d._send = lambda *a, **k: True
        turn = voiced._DictationTurn("d-shut", mock.Mock())
        with mock.patch.object(voiced, "Vad",
                               lambda cfg: mock.Mock(feed=lambda f: "")):
            voiced.Daemon._record(d, turn, capture, engine)
        engine.feed.assert_not_called()

    def test_the_mbrola_fallback_does_not_get_a_fresh_budget(self) -> None:
        # R4 N4. Giving the retry the original budget lets one request take
        # twice as long as it asked for.
        seen = []
        engine = tts_lib.EspeakTts.__new__(tts_lib.EspeakTts)
        engine._cfg = {}
        engine.voice = "en-us"
        engine.rate = 170
        engine._mbrola_ok = True
        engine._mbrola_fallback = True
        engine.mbrola_error = ""

        def _fake_run(text, voice, *, budget=None):
            seen.append(budget)
            if voice.startswith("mb-"):
                import time as _t
                _t.sleep(0.02)                # the failed attempt costs time
                raise tts_lib.TtsError("mbrola voice not installed")
            return b"", 22050

        engine._run = _fake_run
        engine.synth("hello", budget=1.0)
        self.assertEqual(len(seen), 2, seen)
        self.assertEqual(seen[0], 1.0)
        self.assertIsNotNone(seen[1])
        self.assertLess(seen[1], 1.0)         # the retry inherits the remainder

    def test_a_budget_that_expires_mid_recording_refuses_the_turn(self) -> None:
        # Found by re-running R4's mutations after the fix: removing this
        # report SURVIVED all 551 tests. An expired recording must not deliver
        # a `final` transcript as though the turn completed -- and the earlier
        # boundary checks cannot cover it, because the budget is still live
        # when recording STARTS. The clock advances on the first read.
        clock = _Clock()
        turn = voiced._DictationTurn("d-mid", mock.Mock(),
                                     clock() + 10.0, clock)
        capture = mock.Mock()
        capture.frame_bytes = 320
        capture.overruns = 0
        capture.error = ""
        capture.read.side_effect = (
            lambda *a, **k: (clock.advance(20.0), b"\x00" * 320)[1])
        engine = mock.Mock()
        engine.supports_partials = False
        engine.end_utterance.return_value = ""
        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent",
                          "max_seconds": 120},
                  "vad": {"silence_ms": 1}}
        d._stopping = threading.Event()
        d._warn = d._debug = lambda *a, **k: None
        d._send = lambda *a, **k: True
        d._require_capture_consent = lambda resolved=None: None
        with mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture), \
             mock.patch.object(voiced.stt_lib, "make_stt",
                               lambda *a, **k: engine), \
             mock.patch.object(voiced, "Vad",
                               lambda cfg: mock.Mock(feed=lambda f: "")), \
             mock.patch.object(voiced, "clean_for_injection", lambda t: t):
            with self.assertRaises(DeadlineExceeded):
                voiced.Daemon._dictate(d, turn)
        # and nothing was delivered as a completed turn
        self.assertFalse(
            any("final" in str(c) for c in engine.method_calls))

    def test_an_unbounded_caller_still_gets_an_unbounded_fallback(self) -> None:
        seen = []                                                    # control
        engine = tts_lib.EspeakTts.__new__(tts_lib.EspeakTts)
        engine._cfg = {}
        engine.voice = "en-us"
        engine.rate = 170
        engine._mbrola_ok = True
        engine._mbrola_fallback = True
        engine.mbrola_error = ""

        def _fake_run(text, voice, *, budget=None):
            seen.append(budget)
            if voice.startswith("mb-"):
                raise tts_lib.TtsError("nope")
            return b"", 22050

        engine._run = _fake_run
        engine.synth("hello")
        self.assertEqual(seen, [None, None])


if __name__ == "__main__":
    unittest.main()
