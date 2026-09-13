"""Stop versus expiry: the first cause wins, judged on the token's own clock.

reason() used to be a priority -- once stop was set it said "cancelled",
however late the stop came -- while its docstring justified that by order. A
client whose own timer sent stop-dictation as its budget ran out was handed a
final transcript instead of `deadline`. Every clock here is injected, and no
test mixes it with time.monotonic().
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import threading
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_cancel_order", importlib.machinery.SourceFileLoader(
        "kilix_voiced_cancel_order", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol  # noqa: E402
from voicelib.cancel import Cancellation  # noqa: E402


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class ReasonOrderTestCase(unittest.TestCase):

    def test_a_stop_before_the_deadline_stays_cancelled_after_it_passes(self) -> None:
        clock = _Clock()
        token = Cancellation(clock() + 1.0, clock)
        clock.advance(0.5)
        token.set()
        clock.advance(10.0)
        self.assertTrue(token.expired())
        self.assertEqual(token.reason(), "cancelled")

    def test_a_budget_that_ran_out_before_the_stop_is_a_deadline(self) -> None:
        clock = _Clock()
        token = Cancellation(clock() + 1.0, clock)
        clock.advance(2.0)
        token.set()
        self.assertEqual(token.reason(), "deadline")

    def test_a_stop_exactly_at_the_deadline_is_a_deadline(self) -> None:
        clock = _Clock()
        token = Cancellation(clock() + 1.0, clock)
        clock.advance(1.0)
        token.set()
        self.assertEqual(token.reason(), "deadline")

    def test_a_second_stop_does_not_move_when_the_first_arrived(self) -> None:
        clock = _Clock()
        token = Cancellation(clock() + 1.0, clock)
        token.set()
        clock.advance(5.0)
        token.set()
        self.assertEqual(token.reason(), "cancelled")

    def test_clear_forgets_when_the_stop_arrived(self) -> None:
        clock = _Clock()
        token = Cancellation(clock() + 1.0, clock)
        token.set()
        token.clear()
        self.assertIsNone(token.reason())
        clock.advance(5.0)
        token.set()
        self.assertEqual(token.reason(), "deadline")

    def test_with_no_deadline_a_stop_is_always_cancelled(self) -> None:   # control
        token = Cancellation()
        self.assertIsNone(token.reason())
        token.set()
        self.assertEqual(token.reason(), "cancelled")


def _dictation(clock, deadline_in, read, end_utterance):
    """Run one real _run_dictation with an injected clock; return the datagrams."""
    turn = voiced._DictationTurn("listen-1", mock.Mock(),
                                 deadline=clock() + deadline_in, clock=clock)
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
    capture = mock.Mock(rate=16000, frame_bytes=320, overruns=0, error="")
    capture.read.side_effect = lambda *a, **k: read(turn)
    engine = mock.Mock(supports_partials=False)
    engine.feed.return_value = None
    engine.end_utterance.side_effect = end_utterance
    with mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture), \
         mock.patch.object(voiced.stt_lib, "make_stt", lambda *a, **k: engine), \
         mock.patch.object(voiced, "Vad", lambda cfg: types.SimpleNamespace(
             feed=lambda frame: "")):
        voiced.Daemon._run_dictation(d, turn)
    return turn, sent


class DeliveryBoundaryOrderTestCase(unittest.TestCase):

    FRAME = b"\x00" * 320

    def test_stop_sent_as_the_budget_runs_out_yields_deadline_not_final(self) -> None:
        # The exact scenario that was delivering `late words`: the budget is
        # spent on the second frame, and the client's stop arrives after it.
        clock = _Clock()
        reads = []

        def read(turn):
            reads.append(1)
            if len(reads) == 2:
                clock.advance(2.0)
                turn.stop.set()
            return self.FRAME

        turn, sent = _dictation(clock, 1.0, read, lambda: "late words")
        self.assertEqual(turn.stop.reason(), "deadline")
        self.assertEqual(len(sent), 1, sent)
        self.assertNotIn("final", sent[0])
        self.assertEqual(sent[0]["code"], protocol.ERR_DEADLINE)

    def test_stop_before_deadline_then_decoding_outlives_the_budget_yields_deadline(self) -> None:
        clock = _Clock()

        def read(turn):
            turn.stop.set()                  # stop pressed with budget to spare
            return self.FRAME

        def slow_decode():
            clock.advance(10.0)              # final decoding outlives the budget
            return "words the caller gave up on"

        turn, sent = _dictation(clock, 5.0, read, slow_decode)
        self.assertEqual(turn.stop.reason(), "cancelled")   # stop WAS first...
        self.assertEqual(len(sent), 1, sent)                 # ...yet the caller
        self.assertNotIn("final", sent[0])                   # has given up
        self.assertEqual(sent[0]["code"], protocol.ERR_DEADLINE)

    def test_a_budget_spent_during_recording_skips_final_decoding(self) -> None:
        # The stop came first, but the budget ran out before recording
        # returned. The delivery boundary would refuse this turn anyway, so a
        # post-recording check that asked reason() instead of expired() gives
        # the same datagram -- and survived every outcome test for that. It
        # is not the same EFFECT: it goes on to run final decoding for a
        # caller that has given up. Pin that decoding never starts.
        clock = _Clock()
        decoded = []

        def read(turn):
            turn.stop.set()                  # in time...
            clock.advance(10.0)              # ...and then the budget runs out
            return self.FRAME

        turn, sent = _dictation(clock, 5.0, read,
                                lambda: decoded.append(1) or "words")
        self.assertEqual(turn.stop.reason(), "cancelled")
        self.assertEqual([m.get("code") for m in sent], [protocol.ERR_DEADLINE])
        self.assertEqual(decoded, [], "final decoding ran for an abandoned turn")

    def test_stop_before_deadline_decoded_within_budget_delivers_final(self) -> None:
        clock = _Clock()                                                 # control

        def read(turn):
            turn.stop.set()
            return self.FRAME

        turn, sent = _dictation(clock, 5.0, read, lambda: "hello there")
        self.assertEqual(turn.stop.reason(), "cancelled")
        self.assertEqual([m.get("final") for m in sent], ["hello there"])


if __name__ == "__main__":
    unittest.main()
