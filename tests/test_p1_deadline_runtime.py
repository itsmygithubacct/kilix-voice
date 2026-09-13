"""P08/V13 at RUNTIME: an accepted deadline must actually stop work.

The independent review's finding 1: validating a positive integer proves the
input check, not that a positive duration becomes an expired deadline before
work begins. These tests drive a virtual clock so expiry is deterministic and
no real audio, thread or timer is involved.
"""
import importlib.util
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_loader(
    "kilix_voiced", importlib.machinery.SourceFileLoader(
        "kilix_voiced", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class DeadlineRuntimeTestCase(unittest.TestCase):

    def _turn(self, budget_s, clock):
        engine = mock.Mock()
        engine.synth.return_value = (b"\x00\x00", 24000)
        deadline = clock() + budget_s if budget_s is not None else None
        turn = voiced._SpeechTurn("speak-1", ["one", "two"], engine,
                                  deadline, clock)
        return turn, engine

    def test_a_live_budget_does_not_block_work(self) -> None:   # positive control
        clock = _Clock()
        turn, _ = self._turn(5.0, clock)
        self.assertFalse(turn.expired())

    def test_an_elapsed_budget_expires(self) -> None:
        clock = _Clock()
        turn, _ = self._turn(0.001, clock)
        self.assertFalse(turn.expired())
        clock.advance(0.002)
        self.assertTrue(turn.expired())

    def test_no_deadline_never_expires(self) -> None:
        clock = _Clock()
        turn, _ = self._turn(None, clock)
        clock.advance(10_000)
        self.assertFalse(turn.expired())

    def test_expiry_prevents_synthesis_starting(self) -> None:
        # The reviewer's named failing check: the synthesizer must NOT be called.
        clock = _Clock()
        turn, engine = self._turn(0.001, clock)
        clock.advance(0.002)
        daemon = object.__new__(voiced.Daemon)
        self.assertIsNone(voiced.Daemon._synth(daemon, turn, "one"))
        engine.synth.assert_not_called()

    def test_a_live_budget_does_synthesise(self) -> None:       # positive control
        clock = _Clock()
        turn, engine = self._turn(5.0, clock)
        daemon = object.__new__(voiced.Daemon)
        self.assertEqual(voiced.Daemon._synth(daemon, turn, "one"),
                         (b"\x00\x00", 24000))
        engine.synth.assert_called_once_with("one")

    def test_expiry_stops_delivery_not_only_generation(self) -> None:
        # A clip synthesised just before the budget ran out must not be played.
        import threading
        clock = _Clock()
        turn, _ = self._turn(0.001, clock)
        daemon = object.__new__(voiced.Daemon)
        daemon._lock = threading.RLock()
        daemon._speech = turn
        player = mock.Mock()
        self.assertTrue(voiced.Daemon._play_if_current(
            daemon, turn, player, b"\x00\x00", 24000))
        player.play.assert_called_once()
        clock.advance(0.002)
        player.reset_mock()
        self.assertFalse(voiced.Daemon._play_if_current(
            daemon, turn, player, b"\x00\x00", 24000))
        player.play.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class RuntimeAudioAndTranscriptTestCase(unittest.TestCase):
    """Finding 5: the helpers must be reached from the RUNTIME, not only tests."""

    def test_synth_refuses_an_absurd_clip_from_the_engine(self) -> None:
        from voicelib import protocol
        engine = mock.Mock()
        oversized = b"\x00" * (protocol.MAX_AUDIO_BYTES + 2)
        engine.synth.return_value = (oversized, 24000)
        turn = voiced._SpeechTurn("speak-1", ["one"], engine)
        daemon = object.__new__(voiced.Daemon)
        with self.assertRaises(protocol.MessageTooLarge):
            voiced.Daemon._synth(daemon, turn, "one")

    def test_synth_passes_a_normal_clip_through(self) -> None:      # control
        engine = mock.Mock()
        engine.synth.return_value = (b"\x00\x00" * 100, 24000)
        turn = voiced._SpeechTurn("speak-1", ["one"], engine)
        daemon = object.__new__(voiced.Daemon)
        self.assertEqual(voiced.Daemon._synth(daemon, turn, "one"),
                         (b"\x00\x00" * 100, 24000))

    def test_dictation_datagrams_carry_a_segment_id_from_the_runtime(self) -> None:
        # Finding 5 is precisely that the helper existed and the runtime did
        # not call it, so this asserts the CALL SITES. It is a source check and
        # says so: driving the real path needs a microphone. A regex over the
        # call was tried first and matched only as far as the inner
        # clean_for_injection(text) paren -- exact call text is less clever and
        # actually correct.
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        self.assertIn("clean_for_injection(text), turn.id)", source)
        self.assertIn("protocol.dictation_final(text, turn.id)", source)
        # and neither bare form survives anywhere
        self.assertNotIn("protocol.dictation_final(text)", source)

    def test_partial_and_final_of_one_turn_share_a_stable_pairing(self) -> None:
        from voicelib import protocol
        p = protocol.dictation_partial("the qu", "dictate-1")
        f = protocol.dictation_final("the quick", "dictate-1")
        self.assertEqual(p["segment"], f["segment"])
        self.assertIs(p["stable"], False)
        self.assertIs(f["stable"], True)
