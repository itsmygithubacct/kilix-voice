"""Offline tests for voicelib.vad: the frame-clocked speech detector.

Every frame is synthesised with ``struct`` — no microphone, no audio file, no
device.  Because the fed frames are the detector's only clock, "200 ms of
speech" here is exactly ten frames, and every assertion below is arithmetic on
frame counts rather than a wall-clock race.
"""

from __future__ import annotations

import struct
import unittest

from voicelib import events
from voicelib.vad import Vad

RATE = 16000
FRAME_MS = 20
SAMPLES = RATE * FRAME_MS // 1000        # 320 samples, 640 bytes

# Default vocabulary, restated as frame counts so a failure reads as "opened
# after 9 frames" rather than as a millisecond conversion.
START_FRAMES = 200 // FRAME_MS           # vad.start_ms
SUPPRESSED_FRAMES = 350 // FRAME_MS + 1  # vad.suppressed_start_ms, rounded up
END_FRAMES = 900 // FRAME_MS             # vad.silence_ms


def frame(amplitude: int, samples: int = SAMPLES) -> bytes:
    """Return one s16le mono frame of a square wave at ``amplitude``.

    A square wave rather than a sine because its RMS is exactly the amplitude:
    the level a frame produces is then ``amplitude / 32768`` with no rounding,
    which is what lets these tests state thresholds instead of approximating
    them.
    """
    wave = [amplitude, -amplitude] * (samples // 2)
    return struct.pack(f"<{samples}h", *wave)


SILENCE = frame(0)          # a digitally silent source: level 0.0
DITHER = frame(100)         # 0.0031 — below vad.min_rms, so still silence
ROOM = frame(2000)          # 0.061 — above the initial threshold, not speech
VOICE = frame(8000)         # 0.244
SHOUT = frame(20000)        # 0.610


class VadCase(unittest.TestCase):
    """Shared helper: feed frames and hold the detector to its vocabulary."""

    def drive(self, vad: Vad, chunk: bytes, count: int) -> list[str]:
        """Feed ``chunk`` ``count`` times and return the events it produced."""
        seen = []
        for _ in range(count):
            event = vad.feed(chunk)
            self.assertIn(event, events.VAD_EVENTS)
            seen.append(event)
        return seen


class Silence(VadCase):
    """Nothing below the threshold ever opens a turn."""

    def test_digital_silence_produces_no_speech_event(self) -> None:
        vad = Vad()
        seen = self.drive(vad, SILENCE, 500)
        self.assertEqual(set(seen), {events.VAD_SILENCE})
        self.assertEqual(vad.level, 0.0)

    def test_dither_below_min_rms_is_still_silence(self) -> None:
        # A muted source is not perfectly zero; vad.min_rms is what keeps its
        # noise from being heard as a voice however low the learned floor goes.
        vad = Vad()
        self.assertEqual(set(self.drive(vad, DITHER, 500)),
                         {events.VAD_SILENCE})
        self.assertLess(vad.level, vad.threshold)
        self.assertEqual(vad.threshold, 0.010)

    def test_empty_and_odd_frames_are_silence(self) -> None:
        # A truncated final read from the capture pipe must not raise.
        vad = Vad()
        self.assertEqual(vad.feed(b""), events.VAD_SILENCE)
        self.assertEqual(vad.feed(b"\x01"), events.VAD_SILENCE)
        self.assertEqual(vad.level, 0.0)

    def test_a_burst_shorter_than_the_sustain_never_opens(self) -> None:
        vad = Vad()
        seen = self.drive(vad, VOICE, START_FRAMES - 1)
        self.assertEqual(set(seen), {events.VAD_MAYBE})
        self.assertEqual(self.drive(vad, SILENCE, 1), [events.VAD_SILENCE])

    def test_the_sustain_counter_restarts_after_a_quiet_frame(self) -> None:
        vad = Vad()
        self.drive(vad, VOICE, START_FRAMES - 1)
        self.drive(vad, SILENCE, 1)
        seen = self.drive(vad, VOICE, START_FRAMES - 1)
        self.assertNotIn(events.VAD_SPEECH_START, seen)
        self.assertEqual(vad.feed(VOICE), events.VAD_SPEECH_START)


class Speech(VadCase):
    """A sustained burst opens a turn; trailing silence closes it."""

    def test_speech_opens_after_the_sustain_requirement(self) -> None:
        vad = Vad()
        seen = self.drive(vad, VOICE, START_FRAMES)
        self.assertEqual(seen[:-1], [events.VAD_MAYBE] * (START_FRAMES - 1))
        self.assertEqual(seen[-1], events.VAD_SPEECH_START)
        self.assertAlmostEqual(vad.level, 8000 / 32768)

    def test_speech_start_is_reported_exactly_once(self) -> None:
        vad = Vad()
        seen = self.drive(vad, VOICE, START_FRAMES * 4)
        self.assertEqual(seen.count(events.VAD_SPEECH_START), 1)
        self.assertEqual(set(seen[START_FRAMES:]), {events.VAD_SPEECH})

    def test_a_pause_between_words_does_not_end_the_turn(self) -> None:
        vad = Vad()
        self.drive(vad, VOICE, START_FRAMES)
        self.assertEqual(set(self.drive(vad, SILENCE, END_FRAMES - 1)),
                         {events.VAD_SPEECH})
        self.assertEqual(set(self.drive(vad, VOICE, 5)), {events.VAD_SPEECH})
        # The pause counter restarts, so the turn now runs another full window.
        self.assertEqual(set(self.drive(vad, SILENCE, END_FRAMES - 1)),
                         {events.VAD_SPEECH})

    def test_trailing_silence_closes_the_turn(self) -> None:
        vad = Vad()
        self.drive(vad, VOICE, START_FRAMES)
        seen = self.drive(vad, SILENCE, END_FRAMES)
        self.assertEqual(seen[-1], events.VAD_SPEECH_END)
        self.assertEqual(set(seen[:-1]), {events.VAD_SPEECH})

    def test_the_detector_is_reusable_after_a_turn(self) -> None:
        vad = Vad()
        self.drive(vad, VOICE, START_FRAMES)
        self.drive(vad, SILENCE, END_FRAMES)
        self.assertEqual(self.drive(vad, SILENCE, 3),
                         [events.VAD_SILENCE] * 3)
        seen = self.drive(vad, VOICE, START_FRAMES)
        self.assertEqual(seen[-1], events.VAD_SPEECH_START)

    def test_reset_closes_the_turn_but_keeps_the_learned_floor(self) -> None:
        vad = Vad()
        self.drive(vad, SILENCE, 200)
        self.drive(vad, VOICE, START_FRAMES)
        floor = vad.noise_floor
        vad.reset()
        self.assertEqual(vad.noise_floor, floor)
        self.assertEqual(vad.feed(SILENCE), events.VAD_SILENCE)


class Suppressed(VadCase):
    """While the machine is speaking, barge-in must be deliberate."""

    def test_suppression_requires_a_longer_burst(self) -> None:
        vad = Vad()
        vad.set_suppressed(True)
        seen = self.drive(vad, VOICE, SUPPRESSED_FRAMES - 1)
        self.assertEqual(set(seen), {events.VAD_MAYBE})
        self.assertEqual(vad.feed(VOICE), events.VAD_SPEECH_START)

    def test_a_burst_that_would_open_unsuppressed_does_not(self) -> None:
        vad = Vad()
        vad.set_suppressed(True)
        self.assertNotIn(events.VAD_SPEECH_START,
                         self.drive(vad, VOICE, START_FRAMES))

    def test_the_gate_lifts_when_speech_stops(self) -> None:
        vad = Vad()
        vad.set_suppressed(True)
        self.drive(vad, VOICE, START_FRAMES)
        vad.set_suppressed(False)
        # The run already counted still applies: the requirement is what
        # changed, not the history.
        self.assertEqual(vad.feed(VOICE), events.VAD_SPEECH_START)

    def test_suppression_does_not_change_how_a_turn_ends(self) -> None:
        vad = Vad()
        vad.set_suppressed(True)
        self.drive(vad, VOICE, SUPPRESSED_FRAMES)
        seen = self.drive(vad, SILENCE, END_FRAMES)
        self.assertEqual(seen[-1], events.VAD_SPEECH_END)


class Adaptation(VadCase):
    """The floor tracks the room, in both directions."""

    def test_threshold_is_the_floor_times_the_margin(self) -> None:
        vad = Vad({"vad": {"min_rms": 0.0, "margin": 4.0}})
        vad.feed(frame(1000))
        self.assertAlmostEqual(vad.threshold, vad.noise_floor * 4.0)

    def test_the_floor_falls_towards_a_quiet_room(self) -> None:
        vad = Vad()
        start = vad.noise_floor
        self.drive(vad, SILENCE, 50)
        self.assertLess(vad.noise_floor, start)

    def test_a_constantly_noisy_room_is_learned(self) -> None:
        # The documented cost of not calibrating: a room already above the
        # threshold gets one spurious turn, about a floor window long, and is
        # then quiet. What must not happen is latching open for ever.
        vad = Vad()
        seen = self.drive(vad, ROOM, 400)
        self.assertIn(events.VAD_SPEECH_END, seen)
        self.assertEqual(set(seen[-50:]), {events.VAD_SILENCE})
        self.assertGreater(vad.threshold, vad.level)

    def test_a_voice_still_opens_over_a_learned_room(self) -> None:
        vad = Vad()
        self.drive(vad, ROOM, 400)
        seen = self.drive(vad, SHOUT, START_FRAMES)
        self.assertEqual(seen[-1], events.VAD_SPEECH_START)

    def test_speech_does_not_push_the_floor_onto_itself(self) -> None:
        # The window holds the gaps between words as well as the words, and a
        # minimum keeps the floor on the gaps.
        vad = Vad()
        for _ in range(20):
            self.drive(vad, VOICE, 20)
            self.drive(vad, SILENCE, 5)
        self.assertLessEqual(vad.noise_floor, 0.010)


class Configuration(VadCase):
    """Every timing comes from the config, in whole frames."""

    CFG = {"audio": {"frame_ms": 10},
           "vad": {"start_ms": 50, "suppressed_start_ms": 100,
                   "silence_ms": 100}}

    def test_frame_length_and_timings_are_honoured(self) -> None:
        vad = Vad(self.CFG)
        short = frame(8000, RATE * 10 // 1000)
        seen = self.drive(vad, short, 5)
        self.assertEqual(seen[-1], events.VAD_SPEECH_START)
        quiet = frame(0, RATE * 10 // 1000)
        self.assertEqual(self.drive(vad, quiet, 10)[-1],
                         events.VAD_SPEECH_END)

    def test_suppressed_timing_is_honoured(self) -> None:
        vad = Vad(self.CFG)
        vad.set_suppressed(True)
        short = frame(8000, RATE * 10 // 1000)
        self.assertNotIn(events.VAD_SPEECH_START, self.drive(vad, short, 9))
        self.assertEqual(vad.feed(short), events.VAD_SPEECH_START)

    def test_a_partial_frame_of_sustain_rounds_up(self) -> None:
        # 25 ms of 20 ms frames is two frames: the requirement is never
        # rounded down to something shorter than it was configured for.
        vad = Vad({"vad": {"start_ms": 25}})
        self.assertEqual(vad.feed(VOICE), events.VAD_MAYBE)
        self.assertEqual(vad.feed(VOICE), events.VAD_SPEECH_START)

    def test_a_zero_length_requirement_still_needs_one_frame(self) -> None:
        vad = Vad({"vad": {"start_ms": 0, "silence_ms": 0}})
        self.assertEqual(vad.feed(VOICE), events.VAD_SPEECH_START)
        self.assertEqual(vad.feed(SILENCE), events.VAD_SPEECH_END)

    def test_defaults_apply_with_no_config_at_all(self) -> None:
        for cfg in (None, {}, {"vad": {}}):
            with self.subTest(cfg=cfg):
                vad = Vad(cfg)
                self.assertEqual(vad.noise_floor, 0.010)
                self.assertEqual(vad.threshold, 0.030)
                seen = self.drive(vad, VOICE, START_FRAMES)
                self.assertEqual(seen[-1], events.VAD_SPEECH_START)


if __name__ == "__main__":
    unittest.main()
