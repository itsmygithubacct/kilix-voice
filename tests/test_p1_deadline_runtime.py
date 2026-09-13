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


class CaptureConsentGateTestCase(unittest.TestCase):
    """Finding 2: consent must have a PRODUCTION caller on the capture path."""

    def setUp(self) -> None:
        import shutil, tempfile
        self.home = os.path.realpath(tempfile.mkdtemp(prefix="f104-gate-"))
        self.env = mock.patch.dict(
            os.environ, {"HOME": self.home, "KILIX_VOICE_REQUIRE_CONSENT": "1"},
            clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(shutil.rmtree, self.home, True)
        self.daemon = object.__new__(voiced.Daemon)
        self.daemon._cfg = {}

    def _digest(self):
        from voicelib import consent, settings
        return consent.capture_digest(settings.stt_model(), settings.stt_engine())

    def test_without_a_grant_the_microphone_is_not_opened(self) -> None:
        with self.assertRaises(voiced.DaemonError) as caught:
            voiced.Daemon._require_capture_consent(self.daemon)
        self.assertIn("no recorded consent", str(caught.exception))

    def test_with_a_grant_capture_proceeds(self) -> None:        # positive control
        from voicelib import consent
        consent.grant("dictation", self._digest())
        voiced.Daemon._require_capture_consent(self.daemon)      # must not raise

    def test_a_grant_for_another_configuration_does_not_authorise_this_one(self) -> None:
        # S03 at the gate. Written as "grant a DIFFERENT digest" rather than
        # "patch the model": patching settings depends on module-attribute
        # visibility that other tests in the process can perturb, and a test
        # that silently becomes a no-op is worse than none. The assertion below
        # proves the two digests really differ before anything is concluded.
        from voicelib import consent
        other = consent.capture_digest("some-other-model", "vosk")
        self.assertNotEqual(other, self._digest())
        consent.grant("dictation", other)
        with self.assertRaises(voiced.DaemonError) as caught:
            voiced.Daemon._require_capture_consent(self.daemon)
        self.assertIn("no recorded consent", str(caught.exception))

    def test_a_changed_payload_digest_invalidates(self) -> None:
        # S03 again, at the level the digest is built: the same model with a
        # changed installed payload is not what was agreed to.
        from voicelib import consent
        before = consent.capture_digest("m", "vosk", "")
        after = consent.capture_digest("m", "vosk", "a" * 64)
        self.assertNotEqual(before, after)

    def test_the_gate_is_reached_from_dictate(self) -> None:
        # The point of the finding: the helper existed and nothing called it.
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        body = source[source.index("def _dictate(self"):]
        self.assertIn("self._require_capture_consent()",
                      body[:body.index("def _require_capture_consent")])

    def test_the_gate_is_off_unless_explicitly_required(self) -> None:
        # Deployment default: upgrading must not silently break existing callers.
        with mock.patch.dict(os.environ, {"HOME": self.home}, clear=True):
            voiced.Daemon._require_capture_consent(self.daemon)  # must not raise


class DispatchBindingTestCase(unittest.TestCase):
    """R2 survivors: the DISPATCH wiring, not just the mechanism.

    R1's lesson was 'helpers with no callers'. R2 found I had repeated it one
    level up: the turn honoured a deadline it was given, and nothing tested that
    dispatch actually gave it one. Same for the consent CLI.
    """

    def test_dispatch_binds_the_validated_deadline_onto_the_turn(self) -> None:
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        body = source[source.index("deadline_ms = request.get"):]
        head = body[:body.index("turn.thread")]
        # the computed instant must actually be handed to the turn
        self.assertIn("time.monotonic() + deadline_ms / 1000.0", head)
        self.assertIn("chunks, engine, deadline)", head)

    def test_a_turn_built_without_a_deadline_never_expires(self) -> None:
        # The behavioural half: if dispatch stopped passing it, turns would
        # silently never expire. This is what the source check above protects.
        turn = voiced._SpeechTurn("t", ["a"], mock.Mock())
        self.assertIsNone(turn.deadline)
        self.assertFalse(turn.expired())

    def test_dispatch_refuses_an_already_spent_budget(self) -> None:
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        self.assertIn("protocol.ERR_DEADLINE", source)
        block = source[source.index("if turn.expired():"):]
        self.assertIn("elapsed before", block[:600])

    def test_the_consent_cli_is_dispatched(self) -> None:
        # R2 survivor: removing the dispatch left the flags parsed and inert.
        source = open(os.path.join(ROOT, "kilix-stt")).read()
        self.assertIn("if args.grant_consent or args.revoke_consent:", source)
        self.assertIn("_consent_command(args)", source)
        # and it must count as an action, or the interactive fallback swallows it
        self.assertIn("or args.grant_consent or args.revoke_consent", source)

    def test_the_consent_cli_actually_records(self) -> None:
        import json as _json, shutil, subprocess as _sp, tempfile
        home = tempfile.mkdtemp(prefix="f104-cli-")
        self.addCleanup(shutil.rmtree, home, True)
        env = dict(os.environ, HOME=home)
        out = _sp.run([sys.executable, "kilix-stt", "--grant-consent"],
                      cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr[:300])
        self.assertIn("consent recorded", out.stdout)
        probe = _sp.run(
            [sys.executable, "-c",
             "import sys;sys.path.insert(0,'.');"
             "from voicelib import consent, settings;"
             "print(consent.granted('dictation', consent.capture_digest("
             "settings.stt_model(), settings.stt_engine())))"],
            cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(probe.stdout.strip(), "True", probe.stderr[:300])

    def test_engine_identity_is_part_of_the_capture_digest(self) -> None:
        # R2 survivor: dropping engine from the digest left the suite green.
        from voicelib import consent
        self.assertNotEqual(consent.capture_digest("m", "vosk"),
                            consent.capture_digest("m", "vibevoice"))
        self.assertNotEqual(consent.capture_digest("m", "vosk"),
                            consent.capture_digest("other", "vosk"))


class R2DeadlineSurvivorTestCase(unittest.TestCase):
    """deadline_exact_instant and deadline_terminal_report."""

    class _Clock:
        def __init__(self): self.t = 100.0
        def __call__(self): return self.t

    def test_expiry_is_at_the_exact_instant_not_after_it(self) -> None:
        # `>=` vs `>`: one tick either side of the boundary.
        clock = self._Clock()
        turn = voiced._SpeechTurn("t", ["a"], mock.Mock(), 100.5, clock)
        clock.t = 100.499
        self.assertFalse(turn.expired())
        clock.t = 100.5                      # exactly the instant
        self.assertTrue(turn.expired())

    def test_expiry_is_reported_by_the_worker_as_the_terminal_reason(self) -> None:
        # deadline_terminal_report: without this the run ends quietly, or a
        # player error stands in for a deadline the caller set.
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        block = source[source.index("def _run_speech"):]
        block = block[:block.index("def _synth")]
        self.assertIn("if turn.expired():", block)
        self.assertIn("the request deadline elapsed", block)
        # and it must take precedence over the player error, not follow it
        self.assertLess(block.index("if turn.expired():"),
                        block.index("elif player.error:"))


class R2ConsentDurabilityTestCase(unittest.TestCase):
    """consent_fsync: a write error must not leave the lock held or the record gone."""

    def setUp(self) -> None:
        import shutil, tempfile
        self.home = os.path.realpath(tempfile.mkdtemp(prefix="f104-enospc-"))
        self.env = mock.patch.dict(os.environ, {"HOME": self.home}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.addCleanup(shutil.rmtree, self.home, True)

    def test_a_write_failure_keeps_the_previous_record_and_frees_the_lock(self) -> None:
        from voicelib import consent
        good = "a" * 64
        consent.grant("dictation", good)
        real_fsync = os.fsync

        def boom(fd):
            raise OSError(28, "No space left on device")

        with mock.patch.object(os, "fsync", boom):
            with self.assertRaises(OSError):
                consent.grant("dictation", "b" * 64)
        # the earlier grant must survive: a failed write is not a revocation
        self.assertTrue(consent.granted("dictation", good))
        # and the lock must be free, or every later call deadlocks
        consent.grant("other", good)
        self.assertTrue(consent.granted("other", good))
        # no partial temporary left behind
        leftovers = [n for n in os.listdir(os.path.dirname(consent.consent_path()))
                     if n.startswith(".consent-")]
        self.assertEqual(leftovers, [])


class R2Finding1TestCase(unittest.TestCase):
    """R2 finding 1: preparation must be INSIDE the budget, and expiry must
    stop the audio the listener hears, not only generation."""

    def test_the_budget_starts_before_preparation(self) -> None:
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        body = source[source.index("def _op_speak(self, request: dict)"):]
        body = body[:body.index("def ", 10)]
        start = body.index("deadline_ms = request.get")
        # preparation must come AFTER the clock starts, not before it
        for later in ("self._refresh_config()", "tts_lib.make_tts",
                      "self._speech_chunks("):
            with self.subTest(step=later):
                self.assertGreater(body.index(later), start,
                                   f"{later} runs before the budget starts")

    def test_await_clip_stops_on_expiry_not_only_cancellation(self) -> None:
        clock = lambda: clock.t
        clock.t = 100.0
        turn = voiced._SpeechTurn("t", ["a"], mock.Mock(), 100.5, clock)
        player = mock.Mock()
        player.playing = True
        daemon = object.__new__(voiced.Daemon)
        clock.t = 100.6                       # budget spent while the clip plays
        # A mutant that removes the in-loop expiry check makes this spin
        # forever, and a HANGING test is not a failing test -- it stalls the
        # whole suite instead of reporting. Bound it so the mutant fails fast.
        import threading
        result = {}

        def call():
            result["v"] = voiced.Daemon._await_clip(daemon, turn, player)

        with mock.patch.object(voiced, "SPEECH_POLL_S", 0.001):
            worker = threading.Thread(target=call, daemon=True)
            worker.start()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive(),
                         "_await_clip never returned: expiry is not checked "
                         "inside the wait loop")
        self.assertFalse(result["v"])

    def test_await_clip_returns_true_for_a_live_budget(self) -> None:   # control
        clock = lambda: clock.t
        clock.t = 100.0
        turn = voiced._SpeechTurn("t", ["a"], mock.Mock(), 100.5, clock)
        player = mock.Mock()
        player.playing = False
        daemon = object.__new__(voiced.Daemon)
        self.assertTrue(voiced.Daemon._await_clip(daemon, turn, player))
