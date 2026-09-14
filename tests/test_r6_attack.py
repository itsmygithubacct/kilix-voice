"""R6: fresh-seat attack on F104 P1 at 98c5279.

Copy into tests/ of a copy of the subject and run:
    /usr/bin/python3 -B -m unittest tests.test_r6_attack -v

ATTACK_* classes FAIL against 98c5279 and name a live defect.
COVER_* classes PASS against 98c5279 and FAIL under a named mutation that
survives the shipped 560-test suite -- correct code nothing guards.

No network, no microphone, no audio device. Every provider is a throwaway shell
script in a temp dir, reached through the trusted KILIX_PIPER_TTS override.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import stat
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, filename):
    spec = importlib.util.spec_from_loader(
        name, importlib.machinery.SourceFileLoader(name, os.path.join(ROOT, filename)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


voiced = _load("kilix_voiced_r6", "kilix-voiced")

from voicelib import (consent, models, paths, protocol, settings,  # noqa: E402
                      stt as stt_lib, tts as tts_lib)
from voicelib.arbiter import Arbiter  # noqa: E402
from voicelib.cancel import Cancelled, DeadlineExceeded  # noqa: E402


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _script(directory, name, body):
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write("#!/bin/sh\n" + body + "\n")
    os.chmod(path, stat.S_IRWXU)
    return path


def _speak_daemon():
    d = object.__new__(voiced.Daemon)
    d._session_dir = "/tmp"
    d._cfg = {}
    d._refresh_config = lambda: None
    d._touch = lambda: None
    d._speech_chunks = lambda text: []   # a passing probe answers ok, chunks=0
    return d


def _speak(provider, deadline_ms):
    """Drive the REAL PiperTts.check_available -> piper_status through _dispatch."""
    engine = tts_lib.PiperTts(rate=170)
    d = _speak_daemon()
    with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}), \
         mock.patch.object(voiced.tts_lib, "make_tts", lambda *a, **k: engine):
        started = time.monotonic()
        reply = voiced.Daemon._dispatch(d, protocol.encode(
            {"op": "speak", "text": "hi", "deadline_ms": deadline_ms}))
        return reply, time.monotonic() - started


# ==========================================================================
# ATTACK 1 -- the builder's lead, CONFIRMED.
# ==========================================================================

class ATTACK_PiperProviderFailureCodedAsDeadline(unittest.TestCase):
    """98c5279 codes ANY Piper probe failure as ERR_DEADLINE whenever the caller
    budget is under PIPER_STATUS_TIMEOUT_S (5 s), whether or not the budget was
    spent. The provider's own actionable detail ("not installed, run X") is
    replaced by "the request deadline of N ms elapsed", which did not happen."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _assert_not_a_deadline(self, reply, elapsed, deadline_ms, detail):
        self.assertFalse(reply["ok"], reply)
        self.assertLess(elapsed, deadline_ms / 1000.0,
                        "precondition: the budget was NOT spent")
        self.assertNotEqual(
            reply["code"], protocol.ERR_DEADLINE,
            f"a provider failure {elapsed * 1000:.0f} ms into a {deadline_ms} ms "
            f"budget was reported as a deadline: {reply['error']!r}")
        self.assertIn(detail, reply["error"],
                      "the provider's own diagnosis was discarded")

    def test_piper_not_installed_with_a_3000_ms_deadline(self):
        missing = os.path.join(self.tmp.name, "no-such-kilix-piper-tts")
        reply, elapsed = _speak(missing, 3000)
        self._assert_not_a_deadline(reply, elapsed, 3000, "not installed")

    def test_provider_reports_its_model_missing_with_a_4000_ms_deadline(self):
        payload = json.dumps({"model": models.PIPER_KRISTIN_MODEL,
                              "voice": tts_lib.PIPER_VOICE, "installed": False,
                              "detail": "the kristin model files are absent"})
        provider = _script(self.tmp.name, "piper", f"echo '{payload}'")
        reply, elapsed = _speak(provider, 4000)
        self._assert_not_a_deadline(reply, elapsed, 4000, "kristin model files are absent")

    def test_provider_exits_nonzero_with_a_2000_ms_deadline(self):
        provider = _script(self.tmp.name, "piper", "echo 'provider is broken' >&2; exit 3")
        reply, elapsed = _speak(provider, 2000)
        self._assert_not_a_deadline(reply, elapsed, 2000, "exited 3")


class COVER_PiperProbeHonoursTheBudgetEndToEnd(unittest.TestCase):
    """Control for ATTACK 1 (passes on 98c5279 and must keep passing after a fix):
    a provider that really hangs past the budget IS a deadline, and the probe
    returns near the budget, not after its own 5 s ceiling.

    Also the only test in reach that exercises the REAL probe's budget: every
    shipped daemon test replaces check_available with a Mock, so dropping the
    budget inside check_available or piper_status (R6 M25, M26) survives them."""

    def test_a_hung_provider_is_cut_at_the_budget_and_coded_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = _script(tmp, "piper", "exec sleep 8")
            reply, elapsed = _speak(provider, 300)
        self.assertFalse(reply["ok"], reply)
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)
        self.assertLess(elapsed, 1.5, "the probe outlived a 300 ms budget")


# ==========================================================================
# ATTACK 2 -- status accepts deadline_ms and ignores it (the F02 shape).
# ==========================================================================

class ATTACK_StatusIgnoresItsDeadline(unittest.TestCase):
    """validate_request keeps deadline_ms for every op. _op_status never reads
    it, and _tts_status runs the Piper probe with no budget -- up to its own 5 s
    ceiling -- then answers ok:true long after the caller's budget."""

    def test_a_150_ms_status_is_not_answered_ok_after_3_s(self):
        d = object.__new__(voiced.Daemon)
        d._session_dir = "/tmp"
        d._socket_path = "/tmp/control.sock"
        d._cfg = {}
        d._refresh_config = lambda: None
        d._touch = lambda: None
        d._lock = threading.RLock()
        d._started_at = d._last_activity = time.monotonic()
        d._idle_seconds = 300
        d._speech_error, d._speech_error_serial = "", 0
        d._arbiter = mock.Mock(speaking=False, listening=False)
        with tempfile.TemporaryDirectory() as tmp:
            provider = _script(tmp, "piper", "exec sleep 3")
            with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}), \
                 mock.patch.object(voiced.settings, "tts_engine",
                                   lambda path=None: models.TTS_ENGINE_PIPER):
                started = time.monotonic()
                reply = voiced.Daemon._dispatch(d, protocol.encode(
                    {"op": "status", "deadline_ms": 150}))
                elapsed = time.monotonic() - started
        if reply.get("ok"):
            self.assertLess(
                elapsed, 1.0,
                f"status accepted deadline_ms=150 and answered ok:true after "
                f"{elapsed:.2f} s")
        else:
            self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)


# ==========================================================================
# ATTACK 3 -- a final transcript delivered after the budget expired.
# ==========================================================================

def _dictation_daemon(sent, cfg=None):
    d = object.__new__(voiced.Daemon)
    d._cfg = cfg or {"stt": {"engine": "vosk", "model_path": "/nonexistent",
                             "max_seconds": 120}, "vad": {"silence_ms": 900}}
    d._stopping = threading.Event()
    d._warn = d._debug = lambda *a, **k: None
    d._send = lambda receiver, msg: sent.append(msg) or True
    d._clear_dictation = lambda t: None
    d._touch = lambda: None
    d._require_capture_consent = lambda resolved=None: None
    return d


def _capture(frames):
    capture = mock.Mock()
    capture.rate = 16000
    capture.frame_bytes = 320
    capture.overruns = 0
    capture.error = ""
    capture.read.side_effect = lambda *a, **k: frames.pop(0) if frames else None
    return capture


class ATTACK_FinalDeliveredAfterTheDeadline(unittest.TestCase):
    """a36d47f checks for expiry once, right after _record returns. Everything
    after that check -- capture.stop() (terminate + join, up to ~3 s) and
    engine.end_utterance() (final decoding) -- is unbounded, and the `final` is
    then sent with no second look. A budget that expires there delivers the very
    transcript a36d47f says an abandoned request must not receive."""

    def _run(self, expire_in):
        clock = _Clock()
        turn = voiced._DictationTurn("listen-1", mock.Mock(),
                                     deadline=clock() + 5.0, clock=clock)
        sent = []
        d = _dictation_daemon(sent)
        capture = _capture([b"\x00" * 320, b"\x00" * 320])
        events = iter([voiced.events.VAD_SPEECH_START, voiced.events.VAD_SPEECH_END])
        engine = mock.Mock()
        engine.supports_partials = False
        engine.feed.return_value = None

        def end_utterance():
            if expire_in == "end_utterance":
                clock.advance(10.0)          # final decoding outlives the budget
            return "words the caller already gave up on"
        engine.end_utterance.side_effect = end_utterance
        if expire_in == "capture_stop":
            capture.stop.side_effect = lambda: clock.advance(10.0)
        with mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture), \
             mock.patch.object(voiced.stt_lib, "make_stt", lambda *a, **k: engine), \
             mock.patch.object(voiced, "Vad", lambda cfg: types.SimpleNamespace(
                 feed=lambda f: next(events, ""))):
            voiced.Daemon._run_dictation(d, turn)
        self.assertTrue(turn.stop.expired(), "precondition: the budget is spent")
        self.assertFalse(turn.stop.is_set(), "precondition: nobody pressed stop")
        self.assertEqual(len(sent), 1, sent)
        self.assertNotIn("final", sent[0],
                         f"a final transcript was delivered after the budget "
                         f"expired in {expire_in}")
        self.assertEqual(sent[0].get("code"), protocol.ERR_DEADLINE, sent)

    def test_budget_expires_during_final_decoding(self):
        self._run("end_utterance")

    def test_budget_expires_while_the_recorder_shuts_down(self):
        self._run("capture_stop")


# ==========================================================================
# ATTACK 4 -- a consent refusal from a broken record is coded `unavailable`.
# ==========================================================================

class ATTACK_BrokenConsentRecordCodedUnavailable(unittest.TestCase):
    """R4 finding 3 was closed for ONE consent refusal: no grant. When the
    record itself is unreadable, consent.granted raises ConsentError -- a
    ValueError -- which the worker's broad arm codes `unavailable`, the "no
    microphone" code R4 finding 3 was about. The mic stays shut (checked)."""

    def test_a_corrupt_consent_record_is_denied_not_unavailable(self):
        opened = []
        sent = []
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"KILIX_DATA_HOME": tmp,
                                              "KILIX_VOICE_REQUIRE_CONSENT": "1"}):
                os.makedirs(os.path.dirname(consent.consent_path()), exist_ok=True)
                with open(consent.consent_path(), "w") as handle:
                    handle.write("{ this is not json")
                d = _dictation_daemon(sent)
                del d._require_capture_consent           # the REAL gate
                turn = voiced._DictationTurn("listen-1", mock.Mock())
                with mock.patch.object(voiced.audio, "MicCapture",
                                       lambda cfg: opened.append("mic")):
                    voiced.Daemon._run_dictation(d, turn)
        self.assertEqual(opened, [], "property 1 control: the mic stayed shut")
        self.assertEqual(len(sent), 1, sent)
        self.assertIn("consent", sent[0]["error"].lower())
        self.assertEqual(
            sent[0].get("code"), protocol.ERR_DENIED,
            f"the consent gate refused capture and the caller was told "
            f"{sent[0].get('code')!r}")


# ==========================================================================
# ATTACK 5 -- control-path refusals all wear `internal`; ERR_BUSY is dead code.
# ==========================================================================

class ATTACK_ControlPathRefusalsCodedInternal(unittest.TestCase):
    """Not the recorded malformed-request gap (that is validate_request). These
    are WELL-FORMED requests refused by a handler or by _answer, and every one
    reaches reply_error() with no code, i.e. `internal` -- "the daemon has a
    bug". ERR_BUSY and ERR_NOT_FOUND are used nowhere in production, exactly as
    ERR_DENIED was before R4 finding 3."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sock = os.path.join(self.tmp.name, "dictate-1.sock")

    def _daemon(self):
        d = object.__new__(voiced.Daemon)
        d._session_dir = self.tmp.name
        d._lock = threading.RLock()
        d._cfg = {"stt": {"engine": "vosk"}}
        d._refresh_config = lambda: None
        d._touch = lambda: None
        d._warn = d._debug = lambda *a, **k: None
        d._next_turn_id = lambda kind: f"{kind}-9"
        d._speech = None
        d._player = None
        d._speech_error = ""
        return d

    def test_a_second_dictate_while_one_is_running_is_busy(self):
        d = self._daemon()
        d._arbiter = mock.Mock(listening=True)
        reply = voiced.Daemon._dispatch(d, protocol.encode(
            {"op": "dictate", "sock": self.sock}))
        self.assertIn("already running", reply["error"])
        self.assertEqual(reply["code"], protocol.ERR_BUSY, reply)

    def test_speak_while_dictation_holds_the_microphone_is_busy(self):
        d = self._daemon()
        d._arbiter = Arbiter(self.tmp.name)
        d._arbiter.begin_listen("listen-1")
        d._speech_chunks = lambda text: ["one"]
        engine = mock.Mock(model="m", voice="v", rate=170)
        with mock.patch.object(voiced.tts_lib, "make_tts", lambda *a, **k: engine):
            reply = voiced.Daemon._dispatch(d, protocol.encode(
                {"op": "speak", "text": "hello"}))
        self.assertIn("half-duplex", reply["error"])
        self.assertEqual(reply["code"], protocol.ERR_BUSY, reply)

    def test_a_connection_from_another_uid_is_denied(self):
        d = self._daemon()
        d._peer_uid = lambda conn: os.geteuid() + 1
        reply = voiced.Daemon._answer(d, mock.Mock())
        self.assertEqual(reply["code"], protocol.ERR_DENIED, reply)

    def test_an_oversized_request_is_too_large(self):
        d = self._daemon()
        d._peer_uid = lambda conn: os.geteuid()
        conn = mock.Mock()
        conn.recv.return_value = b"x" * (voiced.MAX_REQUEST_BYTES + 1)
        reply = voiced.Daemon._answer(d, conn)
        self.assertEqual(reply["code"], protocol.ERR_TOO_LARGE, reply)

    def test_dictation_switched_off_is_unavailable(self):
        d = self._daemon()
        d._cfg = {"stt": {"engine": stt_lib.ENGINE_OFF}}
        d._arbiter = mock.Mock(listening=False)
        reply = voiced.Daemon._dispatch(d, protocol.encode(
            {"op": "dictate", "sock": self.sock}))
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)


# ==========================================================================
# ATTACK 6 -- the recording loop rounds the deadline by a poll interval.
# ==========================================================================

class ATTACK_RecordingRoundsTheDeadline(unittest.TestCase):
    """cancel.py: wait_done "clips the wait to the remaining budget, so expiry
    wakes the caller AT the deadline rather than up to one poll interval past
    it. That is the difference between a deadline the code honours and one it
    rounds." Speech uses wait_done. a36d47f moved _record's guard to done() but
    kept the deadline-blind turn.stop.wait() and an unclipped 200 ms read."""

    def _record_elapsed(self, read):
        turn = voiced._DictationTurn("listen-1", mock.Mock(),
                                     deadline=time.monotonic() + 0.03)
        capture = mock.Mock(frame_bytes=320, overruns=0, error="")
        capture.read.side_effect = read
        engine = mock.Mock(supports_partials=False)
        d = _dictation_daemon([])
        with mock.patch.object(voiced, "Vad",
                               lambda cfg: types.SimpleNamespace(feed=lambda f: "")):
            started = time.monotonic()
            voiced.Daemon._record(d, turn, capture, engine)
            return time.monotonic() - started

    def test_an_ended_stream_wakes_at_the_deadline(self):
        # read() returns at once (EOF sentinel): the loop then waits 200 ms on
        # the deadline-blind stop.wait().
        elapsed = self._record_elapsed(lambda timeout=None: None)
        self.assertLess(elapsed, 0.12, f"a 30 ms budget ended recording "
                                       f"after {elapsed * 1000:.0f} ms")

    def test_a_quiet_microphone_read_is_clipped_to_the_budget(self):
        # read() blocks for its full timeout, as queue.get does when no frame
        # arrives.
        def read(timeout=None):
            time.sleep(timeout or 0)
            return None
        elapsed = self._record_elapsed(read)
        self.assertLess(elapsed, 0.12, f"a 30 ms budget ended recording "
                                       f"after {elapsed * 1000:.0f} ms")


# ==========================================================================
# ATTACK 7 -- the grant command and the gate hash different directories.
# ==========================================================================

class ATTACK_GrantCannotSatisfyTheGateUnderAModelOverride(unittest.TestCase):
    """487d82d (R3 F03) moved the DAEMON's digest to the resolved model
    directory, which honours KILIX_VOICE_MODEL_PATH / stt.model_path. The grant
    command, kilix-stt --grant-consent, still hashes the CATALOGUE directory.
    With the override the status page itself recommends, the only command that
    records consent can never record one the gate accepts. Fails closed, so
    property 1 holds; the refusal names a remedy that cannot work."""

    def _flow(self, use_override):
        stt_tool = _load("kilix_stt_r6", "kilix-stt")
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = os.path.join(tmp, "settings")
            with open(settings_file, "w") as handle:
                handle.write(f"{settings.KEY_STT_ENGINE}=vosk\n")
            env = {"HOME": tmp, "GPU_TERMINAL_SETTINGS_FILE": settings_file,
                   "KILIX_STORAGE_HOME": os.path.join(tmp, "storage"),
                   "KILIX_VOICE_REQUIRE_CONSENT": "1"}
            with mock.patch.dict(os.environ, env):
                model_id = settings.stt_model()
                target = (os.path.join(tmp, "my-model") if use_override
                          else paths.model_dir(model_id))
                for rel in models.REQUIRED_FILES["vosk"]:
                    os.makedirs(os.path.dirname(os.path.join(target, rel)), exist_ok=True)
                    with open(os.path.join(target, rel), "wb") as handle:
                        handle.write(b"the model the user has")
                if use_override:
                    os.environ[stt_lib.ENV_MODEL] = target
                stt_tool._consent_command(argparse.Namespace(revoke_consent=False))
                d = object.__new__(voiced.Daemon)
                d._cfg = {"stt": {"engine": settings.stt_engine(), "model": model_id}}
                resolved = stt_lib.resolve_stt(d._cfg)
                self.assertEqual(resolved.model_dir, target)
                voiced.Daemon._require_capture_consent(d, resolved)   # must not raise

    def test_control_catalogue_model_grant_is_accepted(self):
        self._flow(use_override=False)

    def test_override_model_grant_is_accepted(self):
        try:
            self._flow(use_override=True)
        except voiced.ConsentDenied as error:
            self.fail(f"kilix-stt --grant-consent recorded consent and the gate "
                      f"still refused: {error}")


# ==========================================================================
# COVER -- correct code that survives the shipped suite (see mutate_r6.py).
# ==========================================================================

class COVER_StopDuringModelLoadKeepsTheMicShut(unittest.TestCase):
    """R6 M01: the check before capture.start() -- the one its own comment calls
    "the last boundary before the device is opened" -- can be deleted and the
    shipped suite stays green, because every shipped test sets stop BEFORE the
    worker runs, where the first check already catches it."""

    def test_a_stop_that_lands_during_make_stt_never_starts_capture(self):
        sent = []
        d = _dictation_daemon(sent)
        turn = voiced._DictationTurn("listen-1", mock.Mock())
        capture = _capture([])

        def slow_model_load(*a, **k):
            turn.stop.set()                    # stop-dictation arrives mid-load
            return mock.Mock(supports_partials=False)
        with mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture), \
             mock.patch.object(voiced.stt_lib, "make_stt", slow_model_load):
            with self.assertRaises(Cancelled):
                voiced.Daemon._dictate(d, turn)
        capture.start.assert_not_called()


class COVER_AnExpiredRecordingClosesTheMicrophone(unittest.TestCase):
    """R6 M04: on the a36d47f deadline path, DeadlineExceeded is raised BEFORE
    the body's capture.stop(), so the finally's capture.stop() is the only thing
    that ends the recorder. Deleting it survives the shipped suite -- no test
    asserts the recorder is stopped on that path."""

    def test_the_recorder_is_stopped_when_the_budget_expires_mid_recording(self):
        clock = _Clock()
        turn = voiced._DictationTurn("listen-1", mock.Mock(),
                                     deadline=clock() + 10.0, clock=clock)
        capture = _capture([])
        capture.read.side_effect = lambda *a, **k: (clock.advance(20.0), b"\x00" * 320)[1]
        engine = mock.Mock(supports_partials=False)
        d = _dictation_daemon([])
        with mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture), \
             mock.patch.object(voiced.stt_lib, "make_stt", lambda *a, **k: engine), \
             mock.patch.object(voiced, "Vad",
                               lambda cfg: types.SimpleNamespace(feed=lambda f: "")):
            with self.assertRaises(DeadlineExceeded):
                voiced.Daemon._dictate(d, turn)
        capture.start.assert_called_once()
        capture.stop.assert_called()


class COVER_EachPreparationBoundaryIsItsOwnCheck(unittest.TestCase):
    """R6 M02, M03. The builder's R3/F02-no-prep-checks replaced ALL THREE
    turn.stop.check() calls in one mutant, so it proves only that at least one
    is needed. Each deleted alone survives the shipped suite."""

    def test_a_stopped_turn_never_starts_the_consent_hash(self):      # M03
        d = _dictation_daemon([])
        d._require_capture_consent = mock.Mock()
        turn = voiced._DictationTurn("listen-1", mock.Mock())
        turn.stop.set()
        with self.assertRaises(Cancelled):
            voiced.Daemon._dictate(d, turn)
        d._require_capture_consent.assert_not_called()

    def test_a_stop_during_the_consent_hash_never_loads_a_model(self):  # M02
        d = _dictation_daemon([])
        turn = voiced._DictationTurn("listen-1", mock.Mock())
        d._require_capture_consent = lambda resolved=None: turn.stop.set()
        built = []
        with mock.patch.object(voiced.audio, "MicCapture",
                               lambda cfg: built.append("capture") or _capture([])), \
             mock.patch.object(voiced.stt_lib, "make_stt",
                               lambda *a, **k: built.append("model") or mock.Mock()):
            with self.assertRaises(Cancelled):
                voiced.Daemon._dictate(d, turn)
        self.assertEqual(built, [])


class COVER_StopOperationsActuallyStop(unittest.TestCase):
    """R6 M20: _op_stop_dictation can be made a no-op -- turn.stop never set --
    and all 560 shipped tests pass; R5's stop test sets the token by hand.
    R6 M22: _cancel_speech can skip engine.cancel(), leaving a Piper synthesis
    process running for up to its 210 s ceiling after stop-speech or barge-in,
    and all 560 pass."""

    def test_stop_dictation_sets_the_current_turns_token(self):         # M20
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        turn = voiced._DictationTurn("listen-1", mock.Mock())
        d._dictation = turn
        reply = voiced.Daemon._op_stop_dictation(d, {"id": "r"})
        self.assertTrue(reply["stopped"])
        self.assertTrue(turn.stop.is_set())

    def test_stop_speech_cancels_the_engine_in_flight(self):            # M22
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._player = None
        d._arbiter = mock.Mock()
        engine = mock.Mock()
        turn = voiced._SpeechTurn("speak-1", ["one"], engine)
        d._speech = turn
        voiced.Daemon._op_stop_speech(d, {"id": "r"})
        self.assertTrue(turn.cancel.is_set())
        engine.cancel.assert_called_once()


class COVER_BudgetsReachTheProcessesTheyBound(unittest.TestCase):
    """R6 M23, M27, M28. R3 F01's fix is verified at the helper (_bounded) and
    at the keyword (_synth passes budget=). Whether an ENGINE applies it to its
    process, and whether wait_done clips to the deadline, are untested: each
    removal survives the shipped suite. (M25/M26, the probe, are covered by
    COVER_PiperProbeHonoursTheBudgetEndToEnd above.)"""

    def test_wait_done_wakes_at_the_deadline_not_the_timeout(self):     # M23
        token = voiced.Cancellation(time.monotonic() + 0.02)
        started = time.monotonic()
        self.assertTrue(token.wait_done(3.0))
        self.assertLess(time.monotonic() - started, 0.5)

    def test_espeak_synthesis_is_cut_at_the_budget(self):               # M28
        with tempfile.TemporaryDirectory() as tmp:
            engine = tts_lib.EspeakTts({"tts": {"cmd": [_script(tmp, "espeak", "exec sleep 8")]}},
                                       voice="en-us", rate=170)
            started = time.monotonic()
            with self.assertRaises(tts_lib.TtsError):
                engine.synth("hello there", budget=0.3)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_piper_synthesis_is_cut_at_the_budget(self):                # M27
        with tempfile.TemporaryDirectory() as tmp:
            provider = _script(tmp, "piper", "exec sleep 8")
            with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: provider}):
                engine = tts_lib.PiperTts(rate=170)
                started = time.monotonic()
                with self.assertRaises(tts_lib.TtsError):
                    engine.synth("hello there", budget=0.3)
        self.assertLess(time.monotonic() - started, 1.5)


class COVER_DictationSocketUnwoundOnDispatchRefusal(unittest.TestCase):
    """R6 M05: property 4's twin on the dictation side. _op_dictate's
    `except BaseException: receiver.close()` can be deleted and every shipped
    test passes -- an arbiter refusal or a thread that will not start leaks the
    connected dictation socket."""

    def _refused(self, configure):
        sock = mock.Mock()
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._cfg = {"stt": {"engine": "vosk"}}
        d._refresh_config = lambda: None
        d._next_turn_id = lambda kind: f"{kind}-1"
        d._arbiter = mock.Mock(listening=False)
        d._clear_dictation = lambda turn: None
        configure(d)
        with mock.patch.object(voiced, "_connect_dictation", lambda p: sock):
            with self.assertRaises((voiced.ArbiterError, voiced.DaemonError)):
                voiced.Daemon._op_dictate(d, {"id": "r", "sock": "/tmp/s"})
        sock.close.assert_called_once()

    def test_an_arbiter_refusal_closes_the_dictation_socket(self):
        def configure(d):
            d._arbiter.begin_listen.side_effect = voiced.ArbiterError("held")
            d._start = lambda thread, undo: None
        self._refused(configure)

    def test_a_worker_that_will_not_start_closes_the_dictation_socket(self):
        def configure(d):
            d._start = mock.Mock(side_effect=voiced.DaemonError("no thread"))
        self._refused(configure)


class COVER_BroadArmFamiliesStayUnavailable(unittest.TestCase):
    """R6 M34, M35: dropping OSError or ValueError from the worker's broad arm
    sends those families to the `internal` arm; both survive the shipped suite."""

    def _code_for(self, error):
        sent = []
        d = _dictation_daemon(sent)
        d._dictate = mock.Mock(side_effect=error)
        voiced.Daemon._run_dictation(d, voiced._DictationTurn("listen-1", mock.Mock()))
        self.assertEqual(len(sent), 1, sent)
        return sent[0].get("code")

    def test_an_os_error_is_unavailable(self):                           # M35
        self.assertEqual(self._code_for(OSError(24, "Too many open files")),
                         protocol.ERR_UNAVAILABLE)

    def test_a_value_error_is_unavailable(self):                         # M34
        self.assertEqual(self._code_for(ValueError("bad frame")),
                         protocol.ERR_UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
