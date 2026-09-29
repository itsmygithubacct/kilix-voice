"""A start-up pop must not end a dictation turn before anyone speaks.

On the reference laptop the microphone opens with about a second of noise far
above speech level. The energy VAD read it as a speech segment and the
following silence as its end, so every Ctrl+Shift+D turn closed in about three
seconds with nothing recognised (0.2.2 RC3). An engine that reports words now
ends a turn on silence only once it has recognised one.
"""
import threading
import unittest
from types import SimpleNamespace

from tests.test_terminal_outcomes import _LiveTurnsFixture, voiced

LOUD = b"\xff\x3f\x01\xc0" * 160          # one 20 ms frame well above threshold
QUIET = b"\x00\x00" * 320


class FrameCapture:
    """Hands the daemon a fixed frame sequence, then nothing."""

    frame_bytes = 640
    overruns = 0
    error = None
    rate = 16000

    def __init__(self, frames):
        self.frames = list(frames)
        self.read_count = 0

    def read(self, timeout=None):
        if not self.frames:
            return None
        self.read_count += 1
        return self.frames.pop(0)

    def stop(self):
        pass


class WordEngine:
    """Recognises words only in the frames marked as speech."""

    supports_partials = True

    def __init__(self, ends_on_words, speech_from):
        self.ends_on_words = ends_on_words
        self.speech_from = speech_from
        self.fed = 0

    def feed(self, frame):
        self.fed += 1
        return "hello there" if self.fed > self.speech_from else ""


def pop_then_words():
    pop = [LOUD] * 25                        # 0.5 s start-up transient
    gap = [QUIET] * 150                      # 3 s before the user speaks
    speech = [LOUD] * 50                     # 1 s of speech
    tail = [QUIET] * 100                     # 2 s of silence afterwards
    return pop + gap + speech + tail, len(pop) + len(gap)


class DictationWaitsForWordsTests(_LiveTurnsFixture):
    def record(self, ends_on_words):
        frames, speech_from = pop_then_words()
        capture = FrameCapture(frames)
        engine = WordEngine(ends_on_words, speech_from)
        turn = SimpleNamespace(stop=voiced.Cancellation(), progress_lock=threading.Lock(),
                               receiver=None, hold=False, silence_ms=1500, owner="", id="listen-test")
        self.daemon._send = lambda *args, **kwargs: True
        heard = self.daemon._record(turn, capture, engine)
        return heard, engine, speech_from

    def test_a_pop_before_speech_does_not_end_the_turn(self):
        heard, engine, speech_from = self.record(ends_on_words=True)
        self.assertTrue(heard)
        # It listened through the pop and the gap, into the speech.
        self.assertGreater(engine.fed, speech_from + 25)

    def test_an_engine_without_word_reports_keeps_the_old_rule(self):
        heard, engine, speech_from = self.record(ends_on_words=False)
        self.assertTrue(heard)
        self.assertLess(engine.fed, speech_from)     # the pop's silence ended it


class WhichEnginesWaitTests(unittest.TestCase):
    def test_only_the_vosk_recogniser_waits_for_words(self):
        from voicelib import stt
        self.assertIs(getattr(stt.VoskStt, "ends_on_words", False), True)
        # No word reports to wait for: they keep the energy rule.
        self.assertIs(getattr(stt.NullStt, "ends_on_words", False), False)
        self.assertIs(getattr(stt.VibeVoiceStt, "ends_on_words", False), False)

if __name__ == "__main__":
    unittest.main()
