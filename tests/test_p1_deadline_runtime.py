"""P08/V13 at RUNTIME: an accepted deadline must actually stop work.

The independent review's finding 1: validating a positive integer proves the
input check, not that a positive duration becomes an expired deadline before
work begins. These tests drive a virtual clock so expiry is deterministic and
no real audio, thread or timer is involved.
"""
import importlib.util
import os
import threading
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
        # F01: the engine must RECEIVE the remaining budget, not just be
        # called. Without it a 210 s Piper ceiling outlives a 5 s request and
        # no deadline poll can run while communicate() blocks.
        engine.synth.assert_called_once_with("one", budget=5.0)

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
        # Asserted BEHAVIOURALLY. The previous version grepped for the exact
        # string "self._require_capture_consent()", so adding the resolved
        # identity argument broke a test of correct code -- and, worse, it
        # would have passed against a call that was present but unreachable.
        started = []
        daemon = object.__new__(voiced.Daemon)
        daemon._cfg = {"stt": {"engine": "vosk"}}
        refused = voiced.DaemonError("no recorded consent (test)")

        def _refuse(resolved=None):
            raise refused

        daemon._require_capture_consent = _refuse
        turn = voiced._DictationTurn("d-gate", mock.Mock())
        with mock.patch.object(voiced.audio, "MicCapture",
                               lambda cfg: started.append("mic")), \
             mock.patch.object(voiced.stt_lib, "make_stt",
                               lambda *a, **k: started.append("stt")):
            with self.assertRaises(voiced.DaemonError):
                voiced.Daemon._dictate(daemon, turn)
        # Not merely called: called BEFORE anything opens the device.
        self.assertEqual(started, [])

    def test_the_gate_is_ON_by_default(self) -> None:
        # Owner decision 2026-09-13: mandatory, not opt-in. An earlier revision
        # defaulted it off so upgrading would not break existing callers; that
        # meant S02 had no runtime meaning anywhere, which R2 said plainly.
        with mock.patch.dict(os.environ, {"HOME": self.home}, clear=True):
            with self.assertRaises(voiced.DaemonError):
                voiced.Daemon._require_capture_consent(self.daemon)

    def test_it_can_be_disabled_only_deliberately(self) -> None:
        with mock.patch.dict(os.environ,
                             {"HOME": self.home,
                              "KILIX_VOICE_REQUIRE_CONSENT": "0"}, clear=True):
            voiced.Daemon._require_capture_consent(self.daemon)  # must not raise

    def test_the_daemon_never_auto_grants(self) -> None:
        # A grant recorded without asking is not consent, whatever the file
        # says afterwards. The daemon has no terminal; it must refuse, not
        # write a grant on the user's behalf.
        from voicelib import consent
        with mock.patch.dict(os.environ, {"HOME": self.home}, clear=True):
            with self.assertRaises(voiced.DaemonError):
                voiced.Daemon._require_capture_consent(self.daemon)
            self.assertFalse(os.path.exists(consent.consent_path()),
                             "the daemon wrote a consent record by itself")


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
        # HOME alone does NOT isolate the store: KILIX_STORAGE_HOME and
        # KILIX_DATA_HOME override it, and a live Kilix session exports both.
        # An earlier version of this test wrote into the real user store
        # because of that. Strip every stack variable.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("KILIX", "GPU_TERMINAL_", "PLEB_"))}
        env["HOME"] = home
        out = _sp.run([sys.executable, "kilix-stt", "--grant-consent"],
                      cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr[:300])
        self.assertIn("consent recorded", out.stdout)
        probe = _sp.run(
            [sys.executable, "-c",
             "import sys;sys.path.insert(0,'.');"
             "from voicelib import consent, settings;"
             "m=settings.stt_model();e=settings.stt_engine();"
             "print(consent.granted('dictation', consent.capture_digest("
             "m, e, consent.payload_digest(m, e))))"],
            cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(probe.stdout.strip(), "True", probe.stderr[:300])
        # and the record must carry the S02 fields, not just a digest
        import json as _j
        store = _j.loads(open(os.path.join(
            home, ".local/gpu_terminal/kilix/data/voice/consent.json")).read())
        entry = store["grants"]["dictation"]
        for field in ("digest", "granted_utc", "allowed_use",
                      "output_identity", "model_id", "model_revision"):
            self.assertIn(field, entry)

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
        # Behavioural: the old version grepped for "if turn.expired():" and so
        # tested a spelling rather than the report, and would have failed
        # against correct code that phrased the check differently.
        clock = _Clock()
        engine = mock.Mock()
        engine.voice, engine.model, engine.rate, engine.seed = "en-us", "m1", 170, 7
        engine.effective_model = None
        engine.synth.side_effect = lambda text, **kw: (b"\x00\x00", 24000)
        turn = voiced._SpeechTurn("speak-r2", ["one"], engine,
                                  clock() + 0.001, clock)
        turn.receiver = None
        clock.advance(1.0)                       # budget spent
        reports = []
        daemon = object.__new__(voiced.Daemon)
        daemon._lock = threading.RLock()
        daemon._speech = turn
        daemon._warn = lambda *a, **k: None
        daemon._touch = lambda: None
        daemon._arbiter = mock.Mock()
        daemon._report_speech_failure = lambda t, m: reports.append(m)
        player = mock.Mock()
        player.playing = False
        # A player error is ALSO pending: the deadline must win, because the
        # caller's budget is the reason the turn stopped.
        player.error = "the sink went away"
        daemon._get_player = lambda: player
        voiced.Daemon._run_speech(daemon, turn)
        self.assertEqual(len(reports), 1, reports)
        self.assertIn("the request deadline elapsed", reports[0])
        self.assertNotIn("the sink went away", reports[0])


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


class StreamedSynthesisTestCase(unittest.TestCase):
    """A12-A15 at the runtime: chunk descriptors published as clips are queued."""

    def _turn(self, chunks=("one", "two", "three")):
        engine = mock.Mock()
        engine.voice, engine.model, engine.rate, engine.seed = "en-us", "m1", 170, 7
        # A bare Mock auto-creates any attribute, so effective_model would be a
        # Mock object rather than absent -- and synthesis_chunk would refuse it.
        # An engine that reports no effective family sets it to None.
        engine.effective_model = None
        turn = voiced._SpeechTurn("speak-9", list(chunks), engine)
        turn.receiver = mock.Mock()
        return turn

    def _daemon(self):
        import threading
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._warn = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        return d

    def test_a_chunk_is_published_when_the_clip_is_queued(self) -> None:
        global sent; sent = []
        turn = self._turn(); daemon = self._daemon(); daemon._speech = turn
        player = mock.Mock()
        self.assertTrue(voiced.Daemon._play_if_current(
            daemon, turn, player, b"\x00\x00" * 100, 24000))
        player.play.assert_called_once()          # queued...
        self.assertEqual(len(sent), 1)            # ...and published at that moment
        chunk = sent[0]
        self.assertEqual(chunk["sequence"], 0)            # A12
        self.assertEqual(chunk["voice"], "en-us")         # A13
        self.assertEqual(chunk["model"], "m1")            # A13
        self.assertEqual(chunk["seed"], 7)                # A14
        self.assertEqual(chunk["settings"]["rate_wpm"], 170)
        self.assertEqual(chunk["pcm_bytes"], 200)
        self.assertIs(chunk["final"], False)              # A15: more to come

    def test_sequence_advances_and_the_last_chunk_is_final(self) -> None:
        global sent; sent = []
        turn = self._turn(("a", "b")); daemon = self._daemon(); daemon._speech = turn
        for _ in range(2):
            voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        self.assertEqual([c["sequence"] for c in sent], [0, 1])
        self.assertEqual([c["final"] for c in sent], [False, True])

    def test_samples_never_ride_in_the_descriptor(self) -> None:
        global sent; sent = []
        turn = self._turn(); daemon = self._daemon(); daemon._speech = turn
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\xff" * 64, 24000)
        blob = repr(sent[0])
        self.assertNotIn("\\xff", blob)
        self.assertEqual(sent[0]["pcm_bytes"], 64)

    def test_a_caller_that_did_not_ask_gets_nothing(self) -> None:   # control
        global sent; sent = []
        turn = self._turn(); turn.receiver = None
        daemon = self._daemon(); daemon._speech = turn
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        self.assertEqual(sent, [])

    def test_provenance_reports_the_effective_family_not_the_requested_one(self) -> None:
        # R3 F06: eSpeak's MBROLA fallback synthesises through espeak while
        # engine.model still reads "mbrola". Reporting the request would name a
        # path the clip did not take.
        global sent; sent = []
        turn = self._turn(); turn.engine.model = "mbrola"
        turn.engine.effective_model = "espeak"
        daemon = self._daemon(); daemon._speech = turn
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        self.assertEqual(sent[0]["model"], "espeak")

    def test_the_ordinal_tracks_the_clip_not_descriptor_success(self) -> None:
        # R3 F05: a failed descriptor build left the counter behind the real
        # clip position, so later sequences lagged and the true last clip could
        # go unmarked final.
        global sent; sent = []
        turn = self._turn(("a", "b", "c")); daemon = self._daemon(); daemon._speech = turn
        turn.engine.voice = "not a valid token"        # clip 0 descriptor fails
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        turn.engine.voice = "en-us"                    # clips 1 and 2 succeed
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        self.assertEqual([c["sequence"] for c in sent], [1, 2])
        self.assertEqual([c["final"] for c in sent], [False, True])

    def test_a_malformed_descriptor_does_not_take_the_audio_down(self) -> None:
        global sent; sent = []
        turn = self._turn(); turn.engine.voice = "not a valid voice token"
        daemon = self._daemon(); daemon._speech = turn
        player = mock.Mock()
        self.assertTrue(voiced.Daemon._play_if_current(
            daemon, turn, player, b"\x00\x00", 24000))
        player.play.assert_called_once()      # audio still queued
        self.assertEqual(sent, [])            # descriptor dropped, not raised

    def test_failed_descriptor_warning_names_the_clip_ordinal(self) -> None:
        # AUD-07: the warning read turn.sequence AFTER the increment, so a
        # descriptor that failed for clip 0 was logged as chunk 1, and the
        # log never said which turn it belonged to.
        global sent; sent = []
        warnings = []
        turn = self._turn(("a", "b")); daemon = self._daemon(); daemon._speech = turn
        daemon._warn = warnings.append
        turn.engine.voice = "not a valid token"        # clip 0's descriptor fails
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        self.assertEqual(sent, [])
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("chunk 0 of turn speak-9", warnings[0])

    def _fallback_engine(self, tmp):
        """A REAL EspeakTts on the mbrola tier whose mb- voice is not installed."""
        import sys
        from voicelib import tts as tts_lib
        script = os.path.join(tmp, "synth.py")
        with open(script, "w") as handle:
            handle.write(
                "import struct, sys\n"
                "voice = sys.argv[1]\n"
                "sys.stdin.buffer.read()\n"
                "if voice.startswith('mb-'):\n"
                "    sys.exit(1)\n"
                "pcm = b'\\x01\\x00' * 400\n"
                "fmt = struct.pack('<HHIIHH', 1, 1, 22050, 44100, 2, 16)\n"
                "body = (b'fmt ' + struct.pack('<I', 16) + fmt + b'data'\n"
                "        + struct.pack('<I', len(pcm)) + pcm)\n"
                "sys.stdout.buffer.write(b'RIFF' + struct.pack('<I', 4 + len(body))"
                " + b'WAVE' + body)\n")
        # An installed MBROLA voice for en-us, so the tier really tries mb-us1
        # and the fake refuses it: the fallback runs on any host (MB-01).
        share = os.path.join(tmp, "share")
        os.makedirs(os.path.join(share, "mbrola", "us1"))
        open(os.path.join(share, "mbrola", "us1", "us1"), "wb").close()
        with mock.patch.dict(os.environ, {"XDG_DATA_DIRS": share}):
            return tts_lib.EspeakTts(
                {"tts": {"cmd": [sys.executable, "-I", script, "{voice}"]}},
                voice="en-us", rate=170, mbrola=True)

    def test_real_espeak_fallback_reaches_the_descriptor(self) -> None:
        # AUD-04 / V24 A13: the provenance test above passes only because a
        # Mock sets effective_model. This drives a real engine through _synth
        # and _play_if_current: the descriptor must name the path the clip
        # actually took, not the family the settings asked for.
        import tempfile
        global sent; sent = []
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._fallback_engine(tmp)
            turn = voiced._SpeechTurn("speak-9", ["one", "two"], engine)
            turn.receiver = mock.Mock()
            daemon = self._daemon(); daemon._speech = turn
            for chunk in turn.chunks:
                clip = voiced.Daemon._synth(daemon, turn, chunk)
                pcm, rate = clip
                self.assertTrue(voiced.Daemon._play_if_current(
                    daemon, turn, mock.Mock(), pcm, rate,
                    getattr(clip, "provenance", None)))
        self.assertEqual(engine.model, "mbrola")                  # the request
        self.assertEqual([(c["model"], c["voice"]) for c in sent],
                         [("espeak", "en-us"), ("espeak", "en-us")])

    def test_the_descriptor_describes_the_clip_not_the_engines_last_synth(self) -> None:
        # The carried provenance is what the descriptor reports, even when the
        # engine has since synthesised something else -- _run_speech does
        # synthesise the next clip while the current one plays.
        import tempfile
        from voicelib import tts as tts_lib
        global sent; sent = []
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._fallback_engine(tmp)
            turn = voiced._SpeechTurn("speak-9", ["one"], engine)
            turn.receiver = mock.Mock()
            daemon = self._daemon(); daemon._speech = turn
            clip = voiced.Daemon._synth(daemon, turn, "one")
        engine.last_provenance = tts_lib.SynthesisProvenance("mbrola", "mb-en-us")
        pcm, rate = clip
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), pcm, rate,
                                       getattr(clip, "provenance", None))
        self.assertEqual((sent[0]["model"], sent[0]["voice"]), ("espeak", "en-us"))

    def test_the_speech_loop_publishes_what_synth_recorded_for_each_clip(self) -> None:
        # _run_speech happens to publish clip N before it synthesises clip
        # N+1, so reading the engine at publish time used to give the right
        # answer by ordering alone. This engine's recorded provenance moves
        # the moment it has been read once -- as any engine that updates what
        # it reports after a clip would -- so only provenance CARRIED from
        # _synth through the real loop can describe each clip correctly.
        from voicelib import tts as tts_lib

        class DriftingEngine:
            model, voice, rate, seed = "espeak", "en-us", 170, 0

            def __init__(self):
                self.calls = 0
                self.recorded = None

            @property
            def last_provenance(self):
                recorded, self.recorded = self.recorded, None
                return recorded or tts_lib.SynthesisProvenance("mbrola", "moved")

            def synth(self, text, *, budget=None):
                self.calls += 1
                self.recorded = tts_lib.SynthesisProvenance("espeak", f"v{self.calls}")
                return b"\x00\x00", 24000

            def cancel(self):
                pass

        global sent; sent = []
        turn = voiced._SpeechTurn("speak-9", ["one", "two"], DriftingEngine())
        turn.receiver = mock.Mock()
        daemon = self._daemon(); daemon._speech = turn
        daemon._touch = lambda: None
        daemon._arbiter = mock.Mock()
        daemon._report_speech_failure = lambda t, m: None
        player = mock.Mock(playing=False, error="")
        daemon._get_player = lambda: player
        voiced.Daemon._run_speech(daemon, turn)
        self.assertEqual([(c["model"], c["voice"]) for c in sent],
                         [("espeak", "v1"), ("espeak", "v2")])

    def test_the_descriptor_carries_the_engines_seed_and_settings(self) -> None:
        # AUD-05 / V24 A14: a real engine's integer seed and its closed
        # settings reach the descriptor -- the rate it was given, and whether
        # the seed was consumed and the output is reproducible.
        import tempfile
        global sent; sent = []
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._fallback_engine(tmp)
            turn = voiced._SpeechTurn("speak-9", ["one"], engine)
            turn.receiver = mock.Mock()
            daemon = self._daemon(); daemon._speech = turn
            clip = voiced.Daemon._synth(daemon, turn, "one")
            pcm, rate = clip
            voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), pcm, rate,
                                           clip.provenance)
        self.assertEqual(sent[0]["seed"], 0)
        self.assertEqual(sent[0]["settings"], {"turn": "speak-9", "rate_wpm": 170,
                                               "seed_consumed": False,
                                               "reproducible": True})

    def _socket_daemon(self, turn):
        import threading
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._warn = lambda *a, **k: None
        d._speech = turn
        d._player = None
        d._arbiter = mock.Mock()
        return d

    @staticmethod
    def _received_fds(ancillary):
        import socket
        import struct
        size = struct.calcsize("i")
        return [fd for level, kind, blob in ancillary
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS
                for fd in struct.unpack(f"{len(blob) // size}i", blob)]

    def test_descriptor_carries_a_sealed_wav_identical_to_the_queued_clip(self) -> None:
        # AUD-06 / V24 A15: the subscriber can play what the daemon plays.
        import fcntl
        import hashlib
        import socket
        import struct
        from voicelib import audiofd, util
        sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        turn = self._turn(("one", "two"))
        turn.receiver = sender
        daemon = self._socket_daemon(turn)
        player = mock.Mock()
        pcm = bytes(range(256)) * 8 + b"\x07"      # odd: the half sample is dropped
        self.assertTrue(voiced.Daemon._play_if_current(daemon, turn, player, pcm, 22050))
        queued = player.play.call_args.args[0]
        queued = queued[:len(queued) - len(queued) % 2]   # as Player.play trims it
        receiver.settimeout(2)          # a missing descriptor fails; it never hangs
        data, ancillary, _flags, _ = receiver.recvmsg(
            1 << 16, socket.CMSG_SPACE(4 * struct.calcsize("i")))
        fds = self._received_fds(ancillary)
        for fd in fds:
            self.addCleanup(os.close, fd)
        self.assertEqual(len(fds), 1)
        descriptor = voiced.protocol.decode(data)
        wav = os.pread(fds[0], 1 << 20, 0)
        self.assertEqual(util.parse_wav_bytes(wav), (queued, 22050))
        self.assertEqual(descriptor["pcm_bytes"], len(queued))
        self.assertEqual(descriptor["byte_length"], len(wav))
        self.assertEqual(len(wav), len(queued) + 44)
        self.assertEqual(descriptor["sha256"], hashlib.sha256(wav).hexdigest())
        self.assertEqual((descriptor["audio_fd"], descriptor["media_type"]), (0, "audio/wav"))
        with self.assertRaises(OSError):
            os.write(fds[0], b"x")
        self.assertEqual(fcntl.fcntl(fds[0], audiofd.F_GET_SEALS) & audiofd.ALL_SEALS,
                         audiofd.ALL_SEALS)
        self.assertEqual(turn.chunks_published, 1)

    def test_subscriber_that_stops_reading_never_blocks_the_lock(self) -> None:
        import socket
        import time
        sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)
        sender.settimeout(1.0)                     # as _connect_dictation sets it
        turn = self._turn(tuple(f"clip {n}" for n in range(21)))
        turn.receiver = sender
        daemon = self._socket_daemon(turn)
        player = mock.Mock()
        open_before = len(os.listdir("/proc/self/fd"))
        for index in range(20):
            started = time.monotonic()
            self.assertTrue(voiced.Daemon._play_if_current(
                daemon, turn, player, b"\x01\x00" * 8000, 22050))
            elapsed = time.monotonic() - started
            if elapsed > 0.05:
                self.fail(f"clip {index} held the daemon lock for {elapsed:.3f} s")
        # Every sealed descriptor made for a clip was closed on the daemon's side.
        self.assertEqual(len(os.listdir("/proc/self/fd")), open_before)
        self.assertEqual(player.play.call_count, 20)          # the audio went on
        self.assertIs(turn.subscriber_lost, True)
        self.assertEqual(turn.delivery_error, voiced.protocol.ERR_UNAVAILABLE)
        receiver.setblocking(False)
        delivered = 0
        try:
            while receiver.recv(1 << 20):
                delivered += 1
        except BlockingIOError:
            pass
        self.assertLess(delivered, 20, "the subscriber kept being sent to after it stalled")
        self.assertEqual(delivered, turn.chunks_published)
        # Drained now, the lost subscriber is still not sent to: its stream has
        # a gap in it, and resuming would hand it clips out of sequence.
        voiced.Daemon._play_if_current(daemon, turn, player, b"\x01\x00" * 8000, 22050)
        with self.assertRaises(BlockingIOError):
            receiver.recv(1 << 20)

    def test_a_stalled_subscriber_does_not_hold_up_stop_speech(self) -> None:
        import socket
        import threading
        import time
        sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)
        # The subscriber has already stopped reading: its queue is full before
        # the first clip, so the publisher's first send is the one that would
        # wait, holding the daemon lock, while stop-speech asks for it.
        sender.setblocking(False)
        try:
            while True:
                sender.send(b"x" * 4096)
        except BlockingIOError:
            pass
        sender.settimeout(1.0)                     # as _connect_dictation sets it
        turn = self._turn(tuple(f"clip {n}" for n in range(200)))
        turn.receiver = sender
        daemon = self._socket_daemon(turn)
        publishing = threading.Event()

        def publish():
            for _ in range(200):
                publishing.set()
                if not voiced.Daemon._play_if_current(
                        daemon, turn, mock.Mock(), b"\x01\x00" * 8000, 22050):
                    return

        worker = threading.Thread(target=publish, daemon=True)
        worker.start()
        self.assertTrue(publishing.wait(5))
        time.sleep(0.05)                           # inside its first send by now
        started = time.monotonic()
        reply = voiced.Daemon._op_stop_speech(daemon, {"id": "s"})
        elapsed = time.monotonic() - started
        worker.join(10)
        self.assertIs(reply["stopped"], True)
        self.assertLess(elapsed, 0.1, f"stop-speech waited {elapsed:.3f} s behind a stalled subscriber")
        self.assertFalse(worker.is_alive())

    def test_descriptor_allocation_failure_ends_subscription_not_audio(self) -> None:
        global sent; sent = []
        turn = self._turn(("a", "b")); daemon = self._daemon(); daemon._speech = turn
        player = mock.Mock()
        with mock.patch.object(voiced.audiofd, "sealed_readonly",
                               side_effect=voiced.audiofd.DescriptorError("no memory")):
            for _ in range(2):
                self.assertTrue(voiced.Daemon._play_if_current(
                    daemon, turn, player, b"\x00\x00", 24000))
        self.assertEqual(player.play.call_count, 2)          # the audio still plays
        self.assertEqual(sent, [])                            # no descriptor without audio
        self.assertIs(turn.subscriber_lost, True)
        self.assertEqual(turn.delivery_error, voiced.protocol.ERR_UNAVAILABLE)

    def test_speak_validates_the_chunk_socket_inside_the_session(self) -> None:
        from voicelib import protocol
        import tempfile
        session = tempfile.mkdtemp(prefix="f104-chunk-")
        os.chmod(session, 0o700)
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_request(
                {"op": "speak", "text": "hi", "chunk_sock": "/etc/evil.sock"},
                session)

    def test_the_pinned_drop_of_sock_on_speak_is_unchanged(self) -> None:
        from voicelib import protocol
        import tempfile
        session = tempfile.mkdtemp(prefix="f104-chunk2-")
        os.chmod(session, 0o700)
        out = protocol.validate_request(
            {"op": "speak", "text": "hi", "sock": "/etc/evil.sock"}, session)
        self.assertNotIn("sock", out)


class CaptureAccountingTestCase(unittest.TestCase):
    """A06 on the capture side: total bytes are counted and refused, not trimmed."""

    def _daemon(self, cfg=None):
        import threading
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._stopping = threading.Event()
        d._cfg = cfg or {}
        d._send = lambda *a, **k: True
        d._warn = lambda *a, **k: None
        return d

    class _Capture:
        rate = 16000
        error = ""
        overruns = 0
        def __init__(self, frame, count, overruns=0):
            self._f, self._n = frame, count
            self.frame_bytes = len(frame)
            self.overruns = overruns
        def read(self, *a, **k):
            if self._n <= 0:
                return None
            self._n -= 1
            return self._f
        def stop(self): pass

    class _Engine:
        supports_partials = False
        def feed(self, frame): return None
        def start_utterance(self): pass
        def end_utterance(self): return ""

    def test_a_flood_of_capture_bytes_is_refused(self) -> None:
        from voicelib import protocol
        import threading
        turn = voiced._DictationTurn("d1", mock.Mock())
        daemon = self._daemon()
        big = b"\x00" * (1024 * 1024)
        frames = protocol.MAX_AUDIO_BYTES // len(big) + 2
        capture = self._Capture(big, frames)
        with mock.patch.object(voiced, "Vad", lambda cfg: mock.Mock(
                feed=lambda f: None)):
            with self.assertRaises(protocol.MessageTooLarge):
                voiced.Daemon._record(daemon, turn, capture, self._Engine())

    def test_an_ordinary_turn_is_not_refused(self) -> None:      # positive control
        from voicelib import protocol
        turn = voiced._DictationTurn("d2", mock.Mock())
        daemon = self._daemon()
        capture = self._Capture(b"\x00" * 3200, 5)
        with mock.patch.object(voiced, "Vad", lambda cfg: mock.Mock(
                feed=lambda f: None)):
            heard = voiced.Daemon._record(daemon, turn, capture, self._Engine())
        self.assertTrue(heard)

    def test_frames_the_queue_dropped_are_counted_too(self) -> None:
        # R3 F04: the counter sat downstream of a queue that discards the
        # oldest frame under load, so lost audio never reached it. A capture
        # reporting overruns must push the total past the ceiling even though
        # the frames it delivered would not.
        from voicelib import protocol
        turn = voiced._DictationTurn("d3", mock.Mock())
        daemon = self._daemon()
        frame = b"\x00" * (1024 * 1024)
        lost = protocol.MAX_AUDIO_BYTES // len(frame) + 2
        capture = self._Capture(frame, 2, overruns=lost)
        with mock.patch.object(voiced, "Vad", lambda cfg: mock.Mock(
                feed=lambda f: None)):
            with self.assertRaises(protocol.MessageTooLarge):
                voiced.Daemon._record(daemon, turn, capture, self._Engine())

    def test_the_counter_is_reached_from_the_capture_loop(self) -> None:
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        body = source[source.index("def _record(self"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("captured_bytes += len(frame)", body)
        self.assertIn("protocol.check_audio_bytes(captured_bytes)", body)


class TruncatedTranscriptTestCase(unittest.TestCase):
    """R3 F04: a transcript built from audio the queue dropped is not a success."""

    class _Capture:
        rate = 16000
        error = ""
        frame_bytes = 3200
        def __init__(self, overruns): self.overruns = overruns; self._n = 2
        def start(self): pass
        def stop(self): pass
        def read(self, *a, **k):
            if self._n <= 0:
                return None
            self._n -= 1
            return b"\x00" * self.frame_bytes

    class _Engine:
        supports_partials = False
        def start_utterance(self): pass
        def feed(self, frame): return None
        def end_utterance(self): return "some words"
        def close(self): pass

    def _daemon(self):
        import threading
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._stopping = threading.Event()
        d._cfg = {}
        d._warn = lambda *a, **k: None
        d._send = lambda *a, **k: True
        return d

    def _run(self, overruns):
        # The consent gate is default-ON and is reached BEFORE capture, which
        # this test incidentally confirms: without disabling it, _dictate
        # refuses for consent and never gets to the overrun check. That is the
        # correct order; it just is not what this test is about.
        turn = voiced._DictationTurn("d9", mock.Mock())
        capture = self._Capture(overruns)
        # Pin the engine. Without this resolve_stt falls through to whatever
        # is in the DEVELOPER'S real settings file -- which on this machine
        # said vibevoice, so the test failed on the host's configuration
        # rather than on the code. A test must not read the user's settings.
        self._daemon_cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent"}}
        with mock.patch.dict(os.environ,
                             {"KILIX_VOICE_REQUIRE_CONSENT": "0"}), \
             mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture), \
             mock.patch.object(voiced.stt_lib, "make_stt",
                               lambda cfg, rate, **kw: self._Engine()), \
             mock.patch.object(voiced, "Vad", lambda cfg: mock.Mock(
                 feed=lambda f: None)), \
             mock.patch.object(voiced, "clean_for_injection", lambda t: t):
            daemon = self._daemon()
            daemon._cfg = self._daemon_cfg
            return voiced.Daemon._dictate(daemon, turn)

    def test_a_turn_that_lost_frames_is_refused(self) -> None:
        with self.assertRaises(voiced.DaemonError) as caught:
            self._run(overruns=3)
        self.assertIn("dropped 3 frame(s)", str(caught.exception))
        self.assertIn("Nothing was delivered", str(caught.exception))

    def test_a_clean_turn_still_delivers(self) -> None:          # positive control
        self._run(overruns=0)                                     # must not raise


class R3SurvivorTestCase(unittest.TestCase):
    """The eight R3 mutations that survived all 504 tests.

    Every one is a lifetime, ordering or correlation property: the mechanisms
    were each correct and did not agree with one another at their boundaries.
    """

    def _turn(self, chunks=("a", "b", "c")):
        engine = mock.Mock()
        engine.voice, engine.model, engine.rate, engine.seed = "en-us", "m1", 170, 7
        engine.effective_model = None
        t = voiced._SpeechTurn("speak-7", list(chunks), engine)
        t.receiver = mock.Mock()
        return t

    def _daemon(self, turn=None):
        import threading
        d = object.__new__(voiced.Daemon)
        d._lock = threading.RLock()
        d._warn = lambda *a, **k: None
        d._send = lambda receiver, msg: captured.append(msg) or True
        d._speech = turn
        return d

    # M01 -----------------------------------------------------------------
    def test_dispatch_connects_the_requested_chunk_receiver(self) -> None:
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        body = source[source.index("def _op_speak(self, request: dict)"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn('request.get("chunk_sock")', body)
        self.assertIn("_connect_dictation(request[\"chunk_sock\"])", body)

    # M02 -----------------------------------------------------------------
    def test_the_chunk_receiver_is_closed_when_the_turn_ends(self) -> None:
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        body = source[source.index("def _run_speech(self"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("turn.receiver.close()", body)
        # and in the finally, so a failing turn still releases it
        self.assertLess(body.index("finally:"), body.index("turn.receiver.close()"))

    # M03 -----------------------------------------------------------------
    def test_every_chunk_carries_its_turn_identity(self) -> None:
        global captured; captured = []
        turn = self._turn(); daemon = self._daemon(turn)
        voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        self.assertEqual(captured[0]["settings"]["turn"], "speak-7")

    # M04 -----------------------------------------------------------------
    def test_the_clip_is_queued_before_it_is_announced(self) -> None:
        # Announcing first would tell a subscriber audio is playing that the
        # player has not accepted yet.
        order = []
        global captured; captured = []
        turn = self._turn(); daemon = self._daemon(turn)
        daemon._send = lambda r, m: order.append("announce") or True
        player = mock.Mock()
        player.play.side_effect = lambda *a, **k: order.append("queue")
        voiced.Daemon._play_if_current(daemon, turn, player, b"\x00\x00", 24000)
        self.assertEqual(order, ["queue", "announce"])

    # M07 + M09 -----------------------------------------------------------
    def test_every_clip_of_a_turn_is_announced_exactly_once(self) -> None:
        global captured; captured = []
        turn = self._turn(("a", "b", "c")); daemon = self._daemon(turn)
        for _ in range(3):
            voiced.Daemon._play_if_current(daemon, turn, mock.Mock(), b"\x00\x00", 24000)
        self.assertEqual(len(captured), len(turn.chunks))          # M07
        self.assertEqual([c["sequence"] for c in captured], [0, 1, 2])
        self.assertEqual(sum(1 for c in captured if c["final"]), 1)  # M09
        self.assertIs(captured[-1]["final"], True)

    # M10 -----------------------------------------------------------------
    def test_the_recogniser_is_closed_when_dictation_ends(self) -> None:
        source = open(os.path.join(ROOT, "kilix-voiced")).read()
        body = source[source.index("def _dictate(self"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("engine.close()", body)
        self.assertLess(body.index("finally:"), body.index("engine.close()"))

    # M11 -----------------------------------------------------------------
    def test_a_finished_dictation_turn_is_not_retained(self) -> None:
        # Behavioural, not a source grep: the cleanup goes through
        # _clear_dictation, and my first attempt asserted a literal assignment
        # that does not appear -- a test that would have failed on correct code.
        import threading
        daemon = object.__new__(voiced.Daemon)
        daemon._lock = threading.RLock()
        daemon._warn = lambda *a, **k: None
        daemon._touch = lambda: None
        daemon._send = lambda *a, **k: True
        daemon._dictate = lambda turn: None
        daemon._arbiter = mock.Mock()          # _clear_dictation ends the lease
        turn = voiced._DictationTurn("d5", mock.Mock())
        daemon._dictation = turn
        voiced.Daemon._run_dictation(daemon, turn)
        self.assertIsNone(daemon._dictation,
                          "the finished turn is still referenced")


class MultiClipTurnTestCase(unittest.TestCase):
    """R3 M07: no test drove _run_speech's loop, so 'synthesise only the first
    clip' passed the entire suite."""

    def _run(self, chunks):
        import threading
        engine = mock.Mock()
        engine.voice, engine.model, engine.rate, engine.seed = "en-us", "m1", 170, 7
        engine.effective_model = None
        # **kw, not a bare `text`: _synth now passes the remaining budget,
        # and a positional-only fake raised TypeError that _run_speech's
        # broad handler swallowed -- the turn looked like it stopped early.
        engine.synth.side_effect = lambda text, **kw: (b"\x00\x00", 24000)
        turn = voiced._SpeechTurn("speak-m", list(chunks), engine)
        turn.receiver = None
        daemon = object.__new__(voiced.Daemon)
        daemon._lock = threading.RLock()
        daemon._speech = turn
        daemon._warn = lambda *a, **k: None
        daemon._touch = lambda: None
        daemon._arbiter = mock.Mock()
        daemon._report_speech_failure = lambda t, m: None
        player = mock.Mock()
        player.playing = False
        player.error = ""
        daemon._get_player = lambda: player
        voiced.Daemon._run_speech(daemon, turn)
        return engine, player

    def test_every_clip_of_a_turn_is_synthesised(self) -> None:
        engine, player = self._run(("one", "two", "three"))
        self.assertEqual([c.args[0] for c in engine.synth.call_args_list],
                         ["one", "two", "three"])
        self.assertEqual(player.play.call_count, 3)

    def test_a_single_clip_turn_still_works(self) -> None:      # control
        engine, player = self._run(("only",))
        self.assertEqual(engine.synth.call_count, 1)
        self.assertEqual(player.play.call_count, 1)
