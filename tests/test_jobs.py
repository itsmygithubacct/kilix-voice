"""TERM-01: the job outcome and the ledger that keeps it."""

from __future__ import annotations

import threading
import unittest

from voicelib import jobs, protocol


def _outcome(job="speak-1", outcome=jobs.OUTCOME_COMPLETED, code=None,
             kind=jobs.KIND_SPEECH, **fields):
    return jobs.JobOutcome(job, kind, outcome, code, **fields)


class OutcomeConsistency(unittest.TestCase):

    def test_outcome_code_consistency_enforced(self) -> None:
        refused = (
            (jobs.OUTCOME_COMPLETED, protocol.ERR_BUSY),
            (jobs.OUTCOME_CANCELLED, protocol.ERR_UNAVAILABLE),
            (jobs.OUTCOME_CANCELLED, None),
            (jobs.OUTCOME_DEADLINE, None),
            (jobs.OUTCOME_DEADLINE, protocol.ERR_CANCELLED),
            (jobs.OUTCOME_FAILED, None),
            (jobs.OUTCOME_FAILED, protocol.ERR_CANCELLED),
            (jobs.OUTCOME_FAILED, protocol.ERR_DEADLINE),
            (jobs.OUTCOME_FAILED, "not-a-code"),
            ("finished", None),
        )
        for outcome, code in refused:
            with self.subTest(outcome=outcome, code=code):
                with self.assertRaises(ValueError):
                    _outcome(outcome=outcome, code=code)
        with self.assertRaises(ValueError):
            _outcome(kind="transcription")
        _outcome(outcome=jobs.OUTCOME_CANCELLED, code=protocol.ERR_CANCELLED)
        _outcome(outcome=jobs.OUTCOME_DEADLINE, code=protocol.ERR_DEADLINE)
        for code in set(protocol.ERROR_CODES) - {protocol.ERR_CANCELLED, protocol.ERR_DEADLINE}:
            _outcome(outcome=jobs.OUTCOME_FAILED, code=code)

    def test_the_record_carries_only_what_is_known(self) -> None:
        self.assertEqual(_outcome(chunks_published=3).record(),
                         {"job": "speak-1", "kind": "speech", "state": "settled",
                          "outcome": "completed", "chunks": 3})
        failed = _outcome(outcome=jobs.OUTCOME_FAILED, code=protocol.ERR_UNAVAILABLE,
                          message="x" * 5000, subscriber_lost=True, delivered=False)
        record = failed.record()
        self.assertEqual(record["code"], protocol.ERR_UNAVAILABLE)
        self.assertLessEqual(len(record["error"]), jobs.MAX_MESSAGE_CHARS + 20)
        self.assertIs(record["subscriber_lost"], True)
        self.assertIs(record["delivered"], False)


class Ledger(unittest.TestCase):

    def test_a_begun_job_reads_running_until_it_settles(self) -> None:
        ledger = jobs.JobLedger()
        self.assertIsNone(ledger.get("speak-1"))
        ledger.begin("speak-1", jobs.KIND_SPEECH)
        self.assertEqual(ledger.get("speak-1"),
                         {"job": "speak-1", "kind": "speech", "state": "running"})
        self.assertTrue(ledger.record(_outcome()))
        self.assertEqual(ledger.get("speak-1")["state"], "settled")

    def test_second_record_is_refused_and_changes_nothing(self) -> None:
        ledger = jobs.JobLedger()
        first = _outcome(chunks_published=2)
        self.assertTrue(ledger.record(first))
        self.assertFalse(ledger.record(_outcome(outcome=jobs.OUTCOME_FAILED,
                                                code=protocol.ERR_UNAVAILABLE,
                                                message="late")))
        self.assertEqual(ledger.get("speak-1"), first.record())

    def test_begin_after_settle_is_noop(self) -> None:
        ledger = jobs.JobLedger()
        ledger.record(_outcome())
        ledger.begin("speak-1", jobs.KIND_SPEECH)
        self.assertEqual(ledger.get("speak-1")["state"], "settled")

    def test_eviction_keeps_running_entries(self) -> None:
        ledger = jobs.JobLedger(max_entries=2)
        ledger.begin("speak-1", jobs.KIND_SPEECH)
        for job in ("speak-2", "speak-3", "speak-4"):
            ledger.record(_outcome(job=job))
        self.assertEqual(ledger.get("speak-1")["state"], "running")
        self.assertIsNone(ledger.get("speak-2"))
        self.assertIsNone(ledger.get("speak-3"))
        self.assertEqual(ledger.get("speak-4")["state"], "settled")

    def test_running_jobs_are_never_evicted_even_over_the_bound(self) -> None:
        ledger = jobs.JobLedger(max_entries=1)
        ledger.begin("speak-1", jobs.KIND_SPEECH)
        ledger.begin("listen-2", jobs.KIND_DICTATION)
        self.assertEqual([r["job"] for r in ledger.recent()], ["speak-1", "listen-2"])

    def test_recent_is_newest_last_and_bounded(self) -> None:
        ledger = jobs.JobLedger()
        for n in range(1, 21):
            ledger.record(_outcome(job=f"speak-{n}"))
        self.assertEqual([r["job"] for r in ledger.recent(3)],
                         ["speak-18", "speak-19", "speak-20"])
        self.assertEqual(len(ledger.recent()), jobs.RECENT)

    def test_delivery_is_marked_once_on_a_settled_job_only(self) -> None:
        ledger = jobs.JobLedger()
        ledger.begin("listen-1", jobs.KIND_DICTATION)
        ledger.mark_delivered("listen-1", True)              # not settled: nothing
        self.assertNotIn("delivered", ledger.get("listen-1"))
        ledger.record(_outcome(job="listen-1", kind=jobs.KIND_DICTATION))
        ledger.mark_delivered("listen-1", False)
        ledger.mark_delivered("listen-1", True)              # already known
        self.assertIs(ledger.get("listen-1")["delivered"], False)

    def test_concurrent_records_leave_exactly_one_outcome(self) -> None:
        ledger = jobs.JobLedger()
        start = threading.Barrier(16)
        results = []

        def settle(index: int) -> None:
            start.wait()
            results.append(ledger.record(_outcome(
                outcome=jobs.OUTCOME_FAILED, code=protocol.ERR_UNAVAILABLE,
                message=f"writer {index}")))

        threads = [threading.Thread(target=settle, args=(i,)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 15)


class TerminalMessage(unittest.TestCase):
    """protocol.job_terminal and declared_minor, the stream's side of P11."""

    def test_the_ledger_vocabulary_is_the_protocols(self) -> None:
        self.assertIs(jobs.OUTCOMES, protocol.JOB_OUTCOMES)

    def test_a_completed_job_carries_no_code_or_error(self) -> None:
        self.assertEqual(protocol.job_terminal(_outcome(chunks_published=4)),
                         {"terminal": True, "job": "speak-1", "kind": "speech",
                          "outcome": "completed", "chunks": 4})

    def test_a_failure_carries_its_code_its_prose_and_a_lost_subscriber(self) -> None:
        message = protocol.job_terminal(_outcome(
            outcome=jobs.OUTCOME_FAILED, code=protocol.ERR_UNAVAILABLE,
            message="playback command exited 3", subscriber_lost=True,
            chunks_published=1))
        self.assertEqual(message["code"], protocol.ERR_UNAVAILABLE)
        self.assertEqual(message["error"], "playback command exited 3")
        self.assertIs(message["subscriber_lost"], True)
        self.assertEqual(protocol.decode(protocol.encode(message)), message)

    def test_an_ending_without_prose_still_says_how_it_ended(self) -> None:
        message = protocol.job_terminal(_outcome(
            outcome=jobs.OUTCOME_CANCELLED, code=protocol.ERR_CANCELLED))
        self.assertEqual(message["error"], "the job ended: cancelled")

    def test_declared_minor(self) -> None:
        for request, minor in (({}, None), ({"v": "1"}, 0), ({"v": 1}, 0),
                               ({"v": "1.2"}, 2), ({"v": "1.10"}, 10)):
            with self.subTest(request=request):
                self.assertEqual(protocol.declared_minor(request), minor)


if __name__ == "__main__":
    unittest.main()
