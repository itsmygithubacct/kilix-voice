"""Tests for the mutants that survived the wave-3a verification battery.

Eight survived. Three are killed next to the fixtures they need: the ingest
read's grace (tests/test_ingest_audio.py), the MBROLA voice built on another
language's database (tests/test_mbrola_voice.py), and the scrubbed stack
prefix (tests/test_suite_isolation.py). The other five are killed here. Each
class names its survivor and asserts the effect that mutant changes.
"""

from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_w3a_survivors", importlib.machinery.SourceFileLoader(
        "kilix_voiced_w3a_survivors", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import jobs, protocol, tts as tts_lib  # noqa: E402


class JobTerminalProseTestCase(unittest.TestCase):
    """Survivor R1c: job_terminal's prose skipped the escaping refusals get.

    Speech failure prose quotes what the engine reported, which can be a path
    whose name is not UTF-8 and so holds lone surrogates. Without the escape
    the terminal cannot be encoded, and the subscriber is told `internal` in
    place of the job's own code.
    """

    OUTCOME = jobs.JobOutcome(
        job="speak-2", kind=jobs.KIND_SPEECH, outcome=jobs.OUTCOME_FAILED,
        code=protocol.ERR_UNAVAILABLE,
        message="read-aloud stopped: cannot run '/opt/voice-\udcff/synth'")

    def test_a_non_utf8_name_in_the_failure_keeps_the_terminal_encodable(self) -> None:
        frame = protocol.encode(protocol.job_terminal(self.OUTCOME),
                                limit=protocol.MAX_REPLY_BYTES)
        sent = protocol.decode(frame)
        self.assertEqual(sent["code"], protocol.ERR_UNAVAILABLE)
        self.assertIn("voice-\\udcff", sent["error"])

    def test_the_daemon_sends_that_terminal_with_the_jobs_own_code(self) -> None:
        d = object.__new__(voiced.Daemon)
        d._warn = lambda *a, **k: None
        terminal = voiced.Daemon._speech_terminal(d, self.OUTCOME)
        self.assertEqual((terminal["outcome"], terminal["code"]),
                         (jobs.OUTCOME_FAILED, protocol.ERR_UNAVAILABLE))


class StopDecidesUnderTheSettleLockTestCase(unittest.TestCase):
    """Survivor T1: stop-dictation decided under a lock of its own.

    A final is settled under the turn's settle lock, and stop-dictation decides
    under that same lock whether there is anything left to stop. Decided under
    any other lock, a stop landing while a final is being settled replied
    stopped, and set the abort, for a job whose transcript was delivered.
    """

    def test_a_stop_racing_a_settling_final_waits_for_it_and_stops_nothing(self) -> None:
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        ledger = jobs.JobLedger()
        turn = voiced._DictationTurn("listen-4", mock.Mock())
        turn.on_settle = ledger.record
        d._dictation = turn
        entered, release = threading.Event(), threading.Event()

        def precondition() -> None:
            entered.set()
            release.wait(10)                  # the final is being settled

        settler = threading.Thread(target=turn.settle, daemon=True, args=(
            jobs.JobOutcome(job="listen-4", kind=jobs.KIND_DICTATION,
                            outcome=jobs.OUTCOME_COMPLETED, code=None), precondition))
        settler.start()
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(5))
        replies = []
        stopper = threading.Thread(daemon=True, target=lambda: replies.append(
            voiced.Daemon._op_stop_dictation(d, {"id": "s", "mode": "abort"})))
        stopper.start()
        stopper.join(0.3)
        waited = stopper.is_alive()
        release.set()
        settler.join(5)
        stopper.join(5)
        self.assertEqual(len(replies), 1)
        self.assertIs(replies[0]["stopped"], False, replies[0])
        self.assertEqual((turn.abort, turn.stop.is_set()), (False, False))
        self.assertEqual(ledger.get("listen-4")["outcome"], "completed")
        self.assertTrue(waited, "the stop decided while the final was being settled")


class EmbeddedKeyAheadOfASmallOneTestCase(unittest.TestCase):
    """Survivor S1: only the LAST sorted embedded-audio key was measured.

    The wave-2 test hid a large value behind a small one. This hides it ahead
    of one: 16 KiB under an early key, a small preview under the last.
    """

    def test_a_large_value_ahead_of_a_small_one_is_too_large_on_every_op(self) -> None:
        session = os.path.realpath(tempfile.mkdtemp(prefix="kv-w3as-", dir="/tmp"))
        self.addCleanup(shutil.rmtree, session, True)
        keys = sorted(protocol.EMBEDDED_AUDIO_KEYS)
        small = base64.b64encode(bytes(16)).decode("ascii")
        over = base64.b64encode(bytes(range(256)) * 64).decode("ascii")    # 16 KiB
        bases = {"speak": {"text": "hi"},
                 "dictate": {"sock": os.path.join(session, "dictate-1.sock")}}
        for early in keys[:-1]:
            for op in protocol.OPS:
                with self.subTest(large_under=early, op=op):
                    message = dict(bases.get(op, {}), op=op, **{early: over, keys[-1]: small})
                    with self.assertRaises(protocol.MessageTooLarge) as caught:
                        protocol.validate_request(message, session)
                    self.assertIn(repr(early), str(caught.exception))


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


class FailureOutranksDeadlineTestCase(unittest.TestCase):
    """Survivor S3: a recorded failure yielded to a deadline.

    A speech job settles from the first match: a recorded failure, then a
    stop, then a deadline. The wave-2 test pinned that order against a stop
    only. A synthesis that fails as the budget runs out still failed, and its
    code says why.
    """

    def test_a_synthesis_error_as_the_budget_runs_out_settles_failed_with_its_code(self) -> None:
        now = [100.0]

        def synth(text, budget=None):
            now[0] += 10.0                            # the budget runs out mid-synthesis...
            raise tts_lib.TtsUnsupported("the voice cannot render this text")

        engine = mock.Mock(model="espeak", voice="en-us", rate=170)
        engine.synth.side_effect = synth
        d = _speech_daemon([])
        turn = voiced._SpeechTurn("speak-6", ["One."], engine, deadline=101.0,
                                  clock=lambda: now[0])
        d._speech = turn
        voiced.Daemon._run_speech(d, turn)
        self.assertEqual(turn.cancel.reason(), "deadline")         # ...and it had
        self.assertEqual((turn.outcome.outcome, turn.outcome.code),
                         (jobs.OUTCOME_FAILED, protocol.ERR_UNSUPPORTED))


class CancelledPiperClipClaimsNoSeedTestCase(unittest.TestCase):
    """Survivor P2: a cancelled Piper synthesis recorded seed 0.

    Reported as likely equivalent, and on the speak wire it is: a cancelled
    clip is never published. The engine's provenance record is the public
    account of the last clip, though, which clip_provenance hands to any
    caller, and A14 says no record claims a seed that did not produce its
    clip. No seed reaches Piper, cancelled or not.
    """

    def test_a_cancelled_piper_synthesis_records_no_seed(self) -> None:
        engine = tts_lib.PiperTts(rate=170)

        class Process:
            returncode = None

            def communicate(self, data, timeout):
                engine.cancel()                      # the stop lands mid-synthesis
                self.returncode = -9
                return b"", b""

            def kill(self):
                self.returncode = -9

            def poll(self):
                return self.returncode

        with mock.patch.object(tts_lib, "piper_binary", return_value="/fixed/piper"), \
                mock.patch.object(tts_lib.subprocess, "Popen", return_value=Process()):
            pcm, _rate = engine.synth("Hello Kristin")
        self.assertEqual(pcm, b"")
        provenance = tts_lib.clip_provenance(engine)
        self.assertEqual((provenance.seed, provenance.seed_consumed, provenance.reproducible),
                         (None, False, False))


if __name__ == "__main__":
    unittest.main()
