"""Production admission, settlement and shutdown with owned synthetic children."""
import os
from pathlib import Path
import subprocess
import threading
import time
from unittest import mock

from tests.test_terminal_outcomes import _LiveTurnsFixture, voiced, wait_until
from voicelib import audio, stt


class RegistryTests(_LiveTurnsFixture):
    def setUp(self):
        super().setUp()
        self.daemon._warn = mock.Mock()
        self.daemon._require_capture_consent = lambda resolved=None: None
        self.captures = []
        self.children = []
        self.fail_capture = True
        self.fail_provider = True
        self.capture_attempts = 0
        self.provider_attempts = 0
        original = audio.MicCapture

        def capture(cfg):
            owner = original(cfg)
            stop = owner.stop

            def stopping():
                self.capture_attempts += 1
                if self.fail_capture:
                    raise InterruptedError("private command and heard text")
                return stop()

            owner.stop = stopping
            self.captures.append(owner)
            # Containment uses the original bound method, even under mutants.
            self.addCleanup(stop)
            return owner

        patch = mock.patch.object(voiced.audio, "MicCapture", side_effect=capture)
        patch.start()
        self.addCleanup(patch.stop)

        def record(turn, capture, engine):
            self.children.append(capture._process)
            close = engine.close

            def closing():
                self.provider_attempts += 1
                if self.fail_provider:
                    raise InterruptedError("private decoder arguments")
                return close()

            engine.close = closing
            self.assertTrue(self.call(op="stop-dictation", mode="abort")["stopped"])
            turn.stop.check()

        self.daemon._record = record

    def begin_and_settle(self):
        receiver = self.receiver()
        result = self.call(op="dictate", sock=receiver.path)
        self.assertTrue(result["ok"], result)
        turn = self.daemon._dictation
        # Worker may already have cleared its active slot.
        if turn is not None:
            turn.thread.join(4)
            self.assertFalse(turn.thread.is_alive(), "cleanup retry is not bounded")
        self.assertTrue(wait_until(lambda: self.daemon._dictation is None, 4))
        row = self.settled(result["turn"])
        self.assertEqual(row["outcome"], "cancelled", (row, self.daemon._warn.call_args_list))
        self.assertTrue(wait_until(lambda: receiver.eof, 2))
        self.assertEqual(len(receiver.msgs), 1)
        self.assertEqual(receiver.msgs[0]["code"], "cancelled")
        return result["turn"]

    def retained(self):
        return self.daemon._failed_dictation_cleanup or {}

    def test_retry_after_settlement_reaps_transient_recorder_failure(self):
        self.fail_capture = self.fail_provider = False
        original = audio.kill_group
        calls = []

        def once(child):
            calls.append(child.pid)
            if len(calls) == 1:
                raise InterruptedError("transient before final group signal")
            return original(child)

        with mock.patch.object(audio, "kill_group", side_effect=once):
            job = self.begin_and_settle()
        self.assertGreaterEqual(len(calls), 2, "no production retry after settlement")
        self.assertEqual(self.retained(), {})
        self.assertIsNotNone(self.children[0].returncode)
        self.assertFalse(Path(f"/proc/{self.children[0].pid}").exists())

    def test_admission_partial_retries_and_several_settled_turns(self):
        for cycle in range(4):
            with self.subTest(cycle=cycle):
                self.fail_capture = self.fail_provider = True
                job = self.begin_and_settle()
                expected = {(job, "capture"): self.captures[-1],
                            (job, "provider"): self.stt_engines[-1]}
                self.assertEqual(self.retained(), expected)
                attempts = (self.capture_attempts, self.provider_attempts)
                self.fail_capture = False
                receiver = self.receiver()
                refused = self.call(op="dictate", sock=receiver.path)
                self.assertFalse(refused["ok"], refused)
                self.assertEqual(refused["code"], "busy")
                self.assertIn("cleanup", refused["error"])
                self.assertIn("voice doctor", refused["error"])
                self.assertGreater(self.capture_attempts, attempts[0])
                self.assertGreater(self.provider_attempts, attempts[1])
                self.assertEqual(self.retained(), {(job, "provider"): expected[(job, "provider")]})
                self.assertEqual(len(self.children), cycle + 1, "refusal still opened capture")
                self.assertFalse(Path(f"/proc/{self.children[-1].pid}").exists())
                # Another refusal cannot forget the remaining owner.
                again = self.call(op="dictate", sock=receiver.path)
                self.assertFalse(again["ok"], again)
                self.assertEqual(self.retained(), {(job, "provider"): expected[(job, "provider")]})
                # Admission itself must retry the old provider before starting
                # another turn. That next turn is settled and cleaned too.
                self.fail_provider = False
                self.begin_and_settle()
                self.assertEqual(self.retained(), {})
                # Count includes this additional successful turn.
                self.children.pop()

    def test_status_reports_retained_owners_without_retry_or_private_data(self):
        job = self.begin_and_settle()
        before = self.capture_attempts, self.provider_attempts
        first = self.status()["status"]
        owners = first.get("retained_cleanup", [])
        self.assertEqual({(r["turn"], r["kind"]) for r in owners},
                         {(job, "capture"), (job, "provider")})
        for row in owners:
            self.assertGreaterEqual(row["age_seconds"], 0)
            self.assertEqual(set(row), {"turn", "kind", "age_seconds"})
        second = self.status()["status"].get("retained_cleanup", [])
        self.assertEqual(before, (self.capture_attempts, self.provider_attempts),
                         "status changed device ownership")
        self.assertEqual(len(second), 2)
        self.assertNotIn("private", str(first))

    def test_shutdown_retries_every_retained_resource(self):
        self.begin_and_settle()
        self.fail_capture = self.fail_provider = False
        before = self.capture_attempts, self.provider_attempts
        start = time.monotonic()
        self.daemon._shutdown()
        self.assertLess(time.monotonic() - start, 3)
        self.assertGreater(self.capture_attempts, before[0])
        self.assertGreater(self.provider_attempts, before[1])
        self.assertEqual(self.retained(), {})
        self.assertFalse(Path(f"/proc/{self.children[0].pid}").exists())

    def test_shutdown_reports_unproven_kind_and_group_without_forgetting(self):
        job = self.begin_and_settle()
        expected = self.retained().copy()
        before = self.capture_attempts, self.provider_attempts
        start = time.monotonic()
        self.daemon._shutdown()
        self.assertLess(time.monotonic() - start, 3)
        self.assertGreater(self.capture_attempts, before[0])
        self.assertGreater(self.provider_attempts, before[1])
        self.assertEqual(self.retained(), expected)
        warnings = [c.args[0] for c in self.daemon._warn.call_args_list]
        final = [w for w in warnings if "shutdown" in w and "unproven" in w]
        self.assertTrue(any(job in w and "capture" in w and str(self.children[0].pid) in w
                            for w in final), final)
        self.assertTrue(any(job in w and "provider" in w for w in final), final)
        self.assertNotIn("private", str(warnings))

    def test_blocked_retry_is_bounded_and_still_owned_at_shutdown(self):
        job = self.begin_and_settle()
        entered, release = threading.Event(), threading.Event()
        owner = self.captures[-1]
        original = owner.stop

        def blocked():
            entered.set()
            release.wait(8)
            return original()

        owner.stop = blocked
        self.fail_capture = self.fail_provider = False
        try:
            start = time.monotonic()
            refused = self.call(op="dictate", sock=self.receiver().path)
            self.assertFalse(refused["ok"], refused)
            self.assertTrue(entered.is_set(), "admission never tried cleanup")
            self.assertLess(time.monotonic() - start, 2)
            self.assertEqual(self.retained(), {(job, "capture"): owner})
            start = time.monotonic()
            self.daemon._shutdown()
            self.assertLess(time.monotonic() - start, 3)
            self.assertIs(self.retained().get((job, "capture")), owner)
        finally:
            release.set()
        self.assertTrue(wait_until(lambda: not self.retained(), 3))
