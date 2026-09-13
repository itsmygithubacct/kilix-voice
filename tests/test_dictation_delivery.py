"""No final transcript once the caller's budget has run out, at any step.

R6 finding 2: _dictate checked for expiry once, when recording returned, and
then ran two unbounded steps -- capture.stop() (terminate plus join) and
engine.end_utterance() (final decoding) -- before sending `final` with no
second look. Each test here expires the budget in exactly ONE step, so each
check has a test that fails without it. Clocks are injected throughout.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import threading
import time
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_delivery", importlib.machinery.SourceFileLoader(
        "kilix_voiced_delivery", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol  # noqa: E402

FRAME = b"\x00" * 320


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class DeliveryBoundaryTestCase(unittest.TestCase):

    def run_turn(self, *, read, stop=None, decode=lambda: "hello there",
                 overruns=0, budget=5.0):
        self.clock = clock = _Clock()
        turn = voiced._DictationTurn("listen-1", mock.Mock(),
                                     deadline=clock() + budget, clock=clock)
        sent = []
        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent",
                          "max_seconds": 120}, "vad": {"silence_ms": 900}}
        d._stopping = threading.Event()
        d._warn = d._debug = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda t: None
        d._touch = lambda: None
        d._require_capture_consent = lambda resolved=None: None
        capture = mock.Mock(rate=16000, frame_bytes=320, overruns=overruns, error="")
        capture.read.side_effect = lambda *a, **k: read(turn)
        if stop is not None:
            capture.stop.side_effect = stop
        self.decoded = []
        engine = mock.Mock(supports_partials=False)
        engine.feed.return_value = None
        engine.end_utterance.side_effect = lambda: (self.decoded.append(1), decode())[1]
        with mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture), \
             mock.patch.object(voiced.stt_lib, "make_stt", lambda *a, **k: engine), \
             mock.patch.object(voiced, "Vad", lambda cfg: types.SimpleNamespace(
                 feed=lambda frame: "")):
            voiced.Daemon._run_dictation(d, turn)
        self.capture = capture
        return sent

    @staticmethod
    def stop_on_first_frame(turn):
        turn.stop.set()                      # the user pressed stop in time
        return FRAME

    def test_a_budget_spent_while_recording_is_refused_before_decoding(self) -> None:
        def read(turn):
            self.clock.advance(10.0)
            return FRAME

        sent = self.run_turn(read=read)
        self.assertEqual([m.get("code") for m in sent], [protocol.ERR_DEADLINE])
        self.assertEqual(self.decoded, [])
        self.capture.stop.assert_called()
        # The refusal names the step the budget ran out in. For this check it
        # is the ONLY difference its removal makes: the next check, after the
        # recorder shuts down, refuses the same turn with the same code, and
        # the finally stops the recorder either way. Where the budget went is
        # what a caller tuning deadline_ms needs to know.
        self.assertIn("while recording", sent[0]["error"])

    def test_a_budget_spent_as_an_overrun_recorder_shuts_down_is_a_deadline(self) -> None:
        # Without the check after capture.stop(), the overruns refusal speaks
        # first and a request that ran out of time is told `unavailable`.
        sent = self.run_turn(read=self.stop_on_first_frame, overruns=3,
                             stop=lambda: self.clock.advance(10.0))
        self.assertEqual(len(sent), 1, sent)
        self.assertEqual(sent[0]["code"], protocol.ERR_DEADLINE, sent)
        self.assertEqual(self.decoded, [])
        self.assertIn("while the recorder shut down", sent[0]["error"])

    def test_a_budget_spent_in_final_decoding_delivers_no_final(self) -> None:
        def slow_decode():
            self.clock.advance(10.0)
            return "words nobody is waiting for"

        sent = self.run_turn(read=self.stop_on_first_frame, decode=slow_decode)
        self.assertEqual(len(sent), 1, sent)
        self.assertNotIn("final", sent[0])
        self.assertEqual(sent[0]["code"], protocol.ERR_DEADLINE)
        self.assertIn("before the transcript was ready", sent[0]["error"])

    def test_a_turn_that_heard_nothing_and_ran_out_says_deadline(self) -> None:
        # Checked before the no-audio refusal: an expired turn is not told
        # "no audio arrived from the microphone".
        def silent(turn):
            turn.stop.set()
            return None

        def slow_empty_decode():
            self.clock.advance(10.0)
            return ""

        sent = self.run_turn(read=silent, decode=slow_empty_decode)
        self.assertEqual(len(sent), 1, sent)
        self.assertEqual(sent[0]["code"], protocol.ERR_DEADLINE, sent)

    def test_a_turn_within_its_budget_delivers_its_final(self) -> None:   # control
        sent = self.run_turn(read=self.stop_on_first_frame)
        self.assertEqual([m.get("final") for m in sent], ["hello there"])

    def test_a_silent_turn_within_its_budget_still_says_no_audio(self) -> None:  # control
        def silent(turn):
            turn.stop.set()
            return None

        sent = self.run_turn(read=silent, decode=lambda: "")
        self.assertEqual(len(sent), 1, sent)
        self.assertEqual(sent[0]["code"], protocol.ERR_UNAVAILABLE, sent)
        self.assertIn("no audio", sent[0]["error"])


class RecordingHonoursTheBudgetTestCase(unittest.TestCase):
    """R6 finding 5: recording ends AT the deadline, not a poll interval past it.

    Real time, because what is measured is how long the loop blocks. The
    budget is 40 ms against a 200 ms poll interval, so a rounded deadline and
    an honoured one are far apart.
    """

    BUDGET = 0.04
    BOUND = 0.15

    def record(self, read, deadline=True):
        turn = voiced._DictationTurn(
            "listen-1", mock.Mock(),
            deadline=(time.monotonic() + self.BUDGET) if deadline else None)
        capture = mock.Mock(frame_bytes=320, overruns=0, error="")
        capture.read.side_effect = lambda timeout=None: read(turn, timeout)
        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"max_seconds": 120}, "vad": {"silence_ms": 900}}
        d._stopping = threading.Event()
        d._warn = d._debug = lambda *a, **k: None
        d._send = lambda *a, **k: True
        engine = mock.Mock(supports_partials=False)
        with mock.patch.object(voiced, "Vad", lambda cfg: types.SimpleNamespace(
                feed=lambda frame: "")):
            started = time.monotonic()
            voiced.Daemon._record(d, turn, capture, engine)
            return time.monotonic() - started, capture

    def test_an_ended_stream_stops_waiting_at_the_deadline(self) -> None:
        # read() returns at once: the loop then waits out the rest of the
        # poll interval, and that wait must wake at the deadline.
        elapsed, _capture = self.record(lambda turn, timeout: None)
        self.assertLess(elapsed, self.BOUND,
                        f"a 40 ms budget ended recording after {elapsed:.3f} s")

    def test_a_quiet_microphone_read_is_clipped_to_the_budget(self) -> None:
        def quiet(turn, timeout):
            time.sleep(timeout or 0)         # blocks as queue.get does
            return None

        elapsed, capture = self.record(quiet)
        self.assertLess(elapsed, self.BOUND,
                        f"a 40 ms budget ended recording after {elapsed:.3f} s")
        for call in capture.read.call_args_list:
            self.assertLessEqual(call.kwargs["timeout"], self.BUDGET)

    def test_an_unbounded_turn_still_polls_at_the_capture_interval(self) -> None:
        seen = []                                                       # control

        def one_poll(turn, timeout):
            seen.append(timeout)
            turn.stop.set()                  # stop-dictation ends the loop
            return None

        self.record(one_poll, deadline=False)
        self.assertEqual(seen, [voiced.CAPTURE_POLL_S])


if __name__ == "__main__":
    unittest.main()
