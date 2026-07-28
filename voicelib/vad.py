"""Frame-clocked energy voice activity detector. See DESIGN.md.

The fed frames are the clock: one frame is ``audio.frame_ms`` of audio, so the
state machine is a pure function of what it was given.  A test that feeds ten
seconds of frames in a millisecond gets exactly the behaviour of a live
microphone, and no wall-clock reading can make a run irreproducible.

A frame counts as loud when its RMS exceeds
``max(vad.min_rms, noise_floor * vad.margin)``.  The floor tracks the quietest
frame in the last ``vad.floor_window_ms``, which is what lets the detector work
in a noisy room with no calibration step.  A minimum rather than an average,
and tracked in every state rather than only during silence, because either
shortcut breaks the case that matters: an average is dragged upwards by the
speech it is supposed to be measuring against, and a floor that only learns
during silence can never recover in a room whose noise already reads as speech
— it would latch open and the turn would run to its timeout.  Speech cannot
push the floor up, because a window of real speech always contains the gaps
between words, and those are the room.  Adaptation is asymmetric: the floor
drops towards a quieter room several times faster than it climbs, so a door
slam ages out in a few frames while a fan that switches on is absorbed over a
second or two.  The floor only climbs once the window is full, because the
minimum of a partial window is not yet a minimum: a turn that opens on somebody
already speaking would otherwise pull the floor up onto their voice and go deaf
for the rest of the sentence.  The cost is that a room already louder than the
threshold gets one spurious turn, about a window long, before it is learned.

``set_suppressed(True)`` is the echo gate: while the machine is speaking, the
microphone hears the speakers, so opening speech demands
``vad.suppressed_start_ms`` of sustained level instead of ``vad.start_ms``.
That is the barge-in trigger, and it is deliberately the slower of the two.

Config keys, all optional, as dotted paths into the daemon's config dict:

=========================  ====================================  =========
``audio.frame_ms``         length of one fed frame, in ms         ``20``
``vad.start_ms``           sustained level that opens speech      ``200``
``vad.suppressed_start_ms`` the same while the machine speaks     ``350``
``vad.silence_ms``         trailing silence that closes speech    ``900``
``vad.margin``             threshold as a multiple of the floor   ``3.0``
``vad.min_rms``            hard threshold floor                   ``0.010``
``vad.floor_window_ms``    window the floor minimises over        ``3000``
``vad.floor_fall``         adaptation rate towards a quiet room   ``0.20``
``vad.floor_rise``         adaptation rate towards a noisy room   ``0.04``
=========================  ====================================  =========

``vad.min_rms`` is what keeps a digitally silent stream — a muted source, a
dead device — mute rather than triggering on its dither.
"""

from __future__ import annotations

import collections

from . import events
from .util import cfg_get, rms16

_SILENCE = "silence"
_MAYBE = "maybe"
_SPEECH = "speech"


def _frames(ms: object, frame_ms: int) -> int:
    """Return the whole frames covering ``ms``, at least one."""
    return max(1, -(-int(ms) // frame_ms))  # ceiling division


class Vad:
    """silence -> maybe -> speech state machine over per-frame RMS."""

    def __init__(self, cfg: dict | None = None) -> None:
        frame_ms = max(1, int(cfg_get(cfg, "audio.frame_ms", 20)))
        self._start_frames = _frames(cfg_get(cfg, "vad.start_ms", 200),
                                     frame_ms)
        self._suppressed_start_frames = _frames(
            cfg_get(cfg, "vad.suppressed_start_ms", 350), frame_ms)
        self._end_frames = _frames(cfg_get(cfg, "vad.silence_ms", 900),
                                   frame_ms)
        self._margin = float(cfg_get(cfg, "vad.margin", 3.0))
        self._min_rms = float(cfg_get(cfg, "vad.min_rms", 0.010))
        self._floor_fall = float(cfg_get(cfg, "vad.floor_fall", 0.20))
        self._floor_rise = float(cfg_get(cfg, "vad.floor_rise", 0.04))
        self._window: collections.deque[float] = collections.deque(
            maxlen=_frames(cfg_get(cfg, "vad.floor_window_ms", 3000), frame_ms))
        self._suppressed = False
        # Start by assuming a quiet room: the first frames of a turn are the
        # ones most likely to be somebody talking, so a floor seeded from them
        # would be seeded from speech.
        self._floor = self._min_rms
        self._level = 0.0
        self._state = _SILENCE
        self._run = 0    # consecutive loud frames before speech opens
        self._quiet = 0  # consecutive quiet frames inside speech

    @property
    def level(self) -> float:
        """Normalised RMS of the most recent frame."""
        return self._level

    @property
    def noise_floor(self) -> float:
        """Current estimate of the room's noise level."""
        return self._floor

    @property
    def threshold(self) -> float:
        """Level a frame must exceed to count as loud."""
        return max(self._min_rms, self._floor * self._margin)

    def set_suppressed(self, on: bool) -> None:
        """Gate the detector while the machine itself is speaking."""
        self._suppressed = bool(on)

    def reset(self) -> None:
        """Return to silence between turns; the learned floor is kept.

        The room did not change while the microphone was closed, so throwing
        away the floor would only cost the next turn its first few hundred
        milliseconds re-learning what it already knew.
        """
        self._state = _SILENCE
        self._run = 0
        self._quiet = 0

    def feed(self, frame: bytes) -> str:
        """Consume one frame and return exactly one ``events.VAD_*`` value."""
        self._level = rms16(frame)
        loud = self._level > self.threshold
        self._adapt_floor()

        if self._state == _SPEECH:
            if loud:
                self._quiet = 0
                return events.VAD_SPEECH
            self._quiet += 1
            if self._quiet >= self._end_frames:
                self.reset()
                return events.VAD_SPEECH_END
            # Still inside the trailing-silence window: a pause between words
            # is part of the utterance, not the end of it.
            return events.VAD_SPEECH

        if loud:
            self._run += 1
            need = (self._suppressed_start_frames if self._suppressed
                    else self._start_frames)
            if self._run >= need:
                self._state = _SPEECH
                self._run = 0
                self._quiet = 0
                return events.VAD_SPEECH_START
            self._state = _MAYBE
            return events.VAD_MAYBE

        # A burst shorter than the sustain requirement — a key press, a cough —
        # falls straight back to silence rather than half-opening a turn.
        self._state = _SILENCE
        self._run = 0
        return events.VAD_SILENCE

    def _adapt_floor(self) -> None:
        self._window.append(self._level)
        target = min(self._window)
        if target < self._floor:
            self._floor += self._floor_fall * (target - self._floor)
        elif len(self._window) == self._window.maxlen:
            self._floor += self._floor_rise * (target - self._floor)
