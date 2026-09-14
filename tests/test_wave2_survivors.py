"""Tests for the mutants that survived the wave-2 verification battery.

Ten survived. Three are killed by the commits that fixed their issues: the
access check in the ingest read (AUD-02), and the two stop-dictation mutants
(TERM-03). The other seven are killed here. Each class names its survivor and
asserts the effect that mutant changes, on a path a caller reaches. One was
reported as equivalent, the stat that became an lstat, and it is not: the
check it sits in is a boundary of its own.
"""

from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_w2_survivors", importlib.machinery.SourceFileLoader(
        "kilix_voiced_w2_survivors", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol, tts as tts_lib  # noqa: E402


def _session(test: unittest.TestCase) -> str:
    session = os.path.realpath(tempfile.mkdtemp(prefix="kv-w2s-", dir="/tmp"))
    test.addCleanup(shutil.rmtree, session, True)
    return session


class OwnControlSocketByIdentityTestCase(unittest.TestCase):
    """Survivor v-w1r3-lstat: the identity check used lstat instead of stat.

    It was reported equivalent because validate_request realpaths every
    receiver before the check runs. But the check is a boundary of its own, and
    its contract is file identity. The same user can put a symlink at a path
    after it was validated. A stat follows that link to the control socket and
    refuses; an lstat compares the link itself and lets it through.
    """

    def test_a_symlink_to_the_control_socket_is_refused_by_the_check_itself(self) -> None:
        session = _session(self)
        control = os.path.join(session, "control.sock")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(listener.close)
        listener.bind(control)
        d = object.__new__(voiced.Daemon)
        info = os.stat(control)
        d._socket_dev, d._socket_ino = info.st_dev, info.st_ino
        link = os.path.join(session, "dictate-1.sock")
        os.symlink(control, link)
        with self.assertRaises(voiced.DaemonRefusedReceiver) as caught:
            voiced.Daemon._refuse_own_control_socket(d, link, "sock")
        self.assertEqual(caught.exception.code, protocol.ERR_MALFORMED)
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)    # control
        self.addCleanup(receiver.close)
        receiver.bind(os.path.join(session, "dictate-2.sock"))
        voiced.Daemon._refuse_own_control_socket(
            d, os.path.join(session, "dictate-2.sock"), "sock")


class SynthesisDescriptorCeilingTestCase(unittest.TestCase):
    """Survivor v-aud05-ceiling-65536: the descriptor ceiling was raised.

    The existing test built its over-limit case from the constant it guards,
    so the case moved with the mutant. AUD-05 sets the ceiling at 2048 encoded
    bytes, far below any subscriber's receive buffer. The boundary here is
    found by encoding real descriptors, not by reading the constant.
    """

    CEILING = 2048

    def test_a_descriptor_may_encode_to_2048_bytes_and_not_one_more(self) -> None:
        base = dict(sequence=0, pcm_bytes=3200, sample_rate=22050, voice="en-us",
                    seed=0, rate_wpm=170, turn="speak-1")
        small = len(protocol.encode(protocol.synthesis_chunk(model="m", **base)))
        at = protocol.synthesis_chunk(model="m" * (self.CEILING - small + 1), **base)
        self.assertEqual(len(protocol.encode(at)), self.CEILING)
        with self.assertRaises(protocol.ProtocolError):
            protocol.synthesis_chunk(model="m" * (self.CEILING - small + 2), **base)


class EveryEmbeddedKeyMeasuredTestCase(unittest.TestCase):
    """Survivor v-aud03-first-key-only: only the first sorted key was measured.

    Every test sent one embedded-audio key per request, so a small value under
    the first key could hide 16 KiB under any later one.
    """

    def test_a_large_value_behind_a_small_one_is_too_large_on_every_op(self) -> None:
        session = _session(self)
        keys = sorted(protocol.EMBEDDED_AUDIO_KEYS)
        small = base64.b64encode(bytes(16)).decode("ascii")
        over = base64.b64encode(bytes(range(256)) * 64).decode("ascii")    # 16 KiB
        bases = {"speak": {"text": "hi"},
                 "dictate": {"sock": os.path.join(session, "dictate-1.sock")}}
        for late in keys[1:]:
            for op in protocol.OPS:
                with self.subTest(hidden_under=late, op=op):
                    message = dict(bases.get(op, {}), op=op, **{keys[0]: small, late: over})
                    with self.assertRaises(protocol.MessageTooLarge) as caught:
                        protocol.validate_request(message, session)
                    self.assertIn(repr(late), str(caught.exception))


def _speech_daemon(sent: list):
    d = object.__new__(voiced.Daemon)
    d._lock = threading.RLock()
    d._warn = d._debug = lambda *a, **k: None
    d._touch = lambda: None
    d._arbiter = mock.Mock()
    d._report_speech_failure = lambda turn, message: None
    player = mock.Mock(error="")
    d._get_player = lambda: player
    d._send = lambda receiver, msg: sent.append(msg) or True
    d._await_clip = lambda turn, player: True
    return d


def _engine(synth):
    engine = mock.Mock(model="espeak", voice="en-us", rate=170)
    engine.synth.side_effect = synth
    engine.last_provenance = tts_lib.SynthesisProvenance(
        "espeak", "en-us", seed=0, reproducible=True, rate_wpm=170)
    return engine


def _clip(text, budget=None):
    return b"\x00" * 64, 22050


class LostSubscriberGetsNoTerminalTestCase(unittest.TestCase):
    """Survivor v-term01-terminal-to-lost-subscriber.

    A 1.2 subscriber that has fallen behind or gone is not sent the job's
    terminal: sending it is a blocking send of up to a second, from the
    speech worker, to a receiver the daemon already gave up on.
    """

    def run_turn(self, lose: bool):
        sent = []
        d = _speech_daemon(sent)
        turn = voiced._SpeechTurn("speak-3", ["One.", "Two."], _engine(_clip))
        turn.receiver = mock.Mock()
        turn.terminal_events = True
        d._speech = turn

        def play(t, player, pcm, rate, provenance=None):
            if lose:
                t.subscriber_lost = True
            return True

        d._play_if_current = play
        voiced.Daemon._run_speech(d, turn)
        return sent, turn

    def test_no_terminal_is_sent_to_a_subscriber_already_lost(self) -> None:
        sent, turn = self.run_turn(lose=True)
        self.assertEqual([m for m in sent if m.get("terminal")], [])
        self.assertEqual((turn.outcome.outcome, turn.outcome.subscriber_lost),
                         ("completed", True))

    def test_a_subscriber_still_there_gets_its_terminal(self) -> None:   # control
        sent, _turn = self.run_turn(lose=False)
        self.assertEqual([m.get("outcome") for m in sent if m.get("terminal")],
                         ["completed"])


class FailureOutranksStopTestCase(unittest.TestCase):
    """Survivor v-term01-failure-yields-to-cancel.

    TERM-01 settles a speech job from the first match: a recorded failure,
    then a stop, then a deadline. A synthesis that fails after a stop was
    pressed still failed. Its code says why, instead of disappearing into a
    cancel that a caller reads as a clean stop.
    """

    def test_a_synthesis_error_after_a_stop_settles_failed_with_its_code(self) -> None:
        sent, holder = [], []

        def synth(text, budget=None):
            if len(holder) == 2:
                return _clip(text)
            turn = holder[0]
            turn.cancel.set()                        # the stop lands mid-synthesis...
            raise tts_lib.TtsUnsupported("the voice cannot render this text")

        d = _speech_daemon(sent)
        turn = voiced._SpeechTurn("speak-4", ["One.", "Two."], _engine(synth))
        holder.extend([turn, "first clip"])
        d._speech = turn
        d._play_if_current = lambda t, player, pcm, rate, provenance=None: (
            holder.pop() and True)
        voiced.Daemon._run_speech(d, turn)
        self.assertEqual(turn.cancel.reason(), "cancelled")      # ...and was recorded
        self.assertEqual((turn.outcome.outcome, turn.outcome.code),
                         ("failed", protocol.ERR_UNSUPPORTED))


class DictationErrorSegmentTestCase(unittest.TestCase):
    """Survivor v-term02-constructor-drops-segment.

    The daemon adds the segment itself when a terminal lacks one, so the wire
    looked the same. The constructor's own contract is that the job id given
    to it is carried; nothing tested that.
    """

    def test_the_constructor_carries_the_job_id_it_is_given(self) -> None:
        self.assertEqual(
            protocol.dictation_error("stopped", protocol.ERR_DEADLINE, segment="listen-3"),
            {"error": "stopped", "code": protocol.ERR_DEADLINE, "segment": "listen-3"})
        self.assertEqual(protocol.dictation_error("x"), {"error": "x"})   # bare form


class StopWaitsForAShortFeedTestCase(unittest.TestCase):
    """Survivor v-term04-quiesce-zero: the stop's wait for a feed was zero.

    The slow-feed test patches the wait, and the race test only examines
    replies that say quiesced. A stop that lands during an ordinary feed must
    wait for it, with the daemon's real bound, and reply quiesced.
    """

    FRAME = b"\x00" * 640

    def test_a_stop_during_a_short_feed_waits_for_it_and_is_quiesced(self) -> None:
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._cfg = {"stt": {"max_seconds": 30}, "vad": {"silence_ms": 900}}
        d._stopping = threading.Event()
        d._warn = d._debug = lambda *a, **k: None
        d._send = lambda receiver, msg: True
        turn = voiced._DictationTurn("listen-1", mock.Mock())
        d._dictation = turn
        feeding, feeds_ended = threading.Event(), []

        def feed(frame):
            feeding.set()
            time.sleep(0.15)
            feeds_ended.append(time.monotonic())
            return "word"

        engine = SimpleNamespace(supports_partials=True, feed=feed)
        capture = SimpleNamespace(rate=16000, frame_bytes=len(self.FRAME), overruns=0,
                                  error="", read=lambda timeout=None: self.FRAME)
        with mock.patch.object(voiced, "Vad", lambda cfg: SimpleNamespace(
                feed=lambda frame: "")):
            worker = threading.Thread(target=voiced.Daemon._record,
                                      args=(d, turn, capture, engine), daemon=True)
            worker.start()
            self.assertTrue(feeding.wait(5))
            reply = voiced.Daemon._op_stop_dictation(d, {"id": "s"})
            replied = time.monotonic()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual((reply["stopped"], reply["quiesced"]), (True, True), reply)
        self.assertTrue(feeds_ended and feeds_ended[0] <= replied,
                        "the stop replied before the feed it landed in had ended")


if __name__ == "__main__":
    unittest.main()
