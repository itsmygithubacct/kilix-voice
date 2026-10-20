"""The retained provider keeps the real lease's acknowledgement descriptor."""
import contextlib
import os
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest import mock

from tests.test_accelerator import _LeaseFixture
from tests import test_process_cleanup
from tests.test_terminal_outcomes import _worker_daemon, voiced
from voicelib import accelerator, jobs, stt


class RecoveryLeaseTests(_LeaseFixture):
    def test_stop_during_cleanup_before_provider_construction_releases_unused_grant(self):
        sent, ledger = [], jobs.JobLedger()
        daemon = _worker_daemon(sent, ledger)
        daemon._cfg = {}
        daemon._lock = threading.RLock()
        turn = voiced._DictationTurn("before-provider", mock.Mock())
        turn.on_settle = ledger.record
        daemon._dictation = turn
        capture = mock.Mock()

        def stopping():
            self.assertTrue(daemon._op_stop_dictation({"id": "abort", "mode": "abort"})["stopped"])

        capture.stop.side_effect = stopping
        with self.execution(job_id=turn.id) as grant:
            self.addCleanup(grant._lease.release, cleanup_complete=False)
            with mock.patch.object(grant._lease, "release", wraps=grant._lease.release) as releasing, \
                    mock.patch.object(voiced.audio, "MicCapture", return_value=capture), \
                    mock.patch.object(voiced.stt_lib, "make_stt", side_effect=stt.SttError("construction failed")):
                daemon._dictate = lambda t: daemon._record_turn(t, SimpleNamespace(), grant)
                daemon._run_dictation(turn)
                self.assertEqual([m["code"] for m in sent], ["cancelled"])
                releasing.assert_called_once_with(cleanup_complete=True)

    def test_provider_retry_acknowledges_only_its_proven_real_grant(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            with self.subTest(provider=engine_type.__name__):
                # A quarantining mutant in one provider case must not poison
                # the other provider's independent private namespace.
                self.namespace = os.path.join(self.parent, engine_type.__name__)
                engine, child = test_process_cleanup.CleanupTests.provider(self, engine_type)
                sent, ledger = [], jobs.JobLedger()
                daemon = _worker_daemon(sent, ledger)
                daemon._cfg = {}
                turn = voiced._DictationTurn("retained-grant", mock.Mock())
                turn.on_settle = ledger.record
                daemon._record = mock.Mock(side_effect=stt.SttError("decoder stopped"))
                leases = []
                acquire = self.lease.acquire

                def acquiring(**kwargs):
                    lease = acquire(**kwargs)
                    lease.release = mock.Mock(wraps=lease.release)
                    leases.append(lease)
                    return lease

                def dictate(t):
                    with self.execution(job_id=t.id) as grant:
                        daemon._record_turn(t, SimpleNamespace(), grant)

                daemon._dictate = dictate
                with mock.patch.object(self.lease, "acquire", side_effect=acquiring), \
                        mock.patch.object(voiced.audio, "MicCapture", return_value=mock.Mock()), \
                        mock.patch.object(voiced.stt_lib, "make_stt", return_value=engine), \
                        mock.patch.object(child, "wait", side_effect=subprocess.TimeoutExpired("provider", 5)):
                    daemon._run_dictation(turn)
                self.assertEqual(len(leases), 1, "the private lease was not acquired")
                lease = leases[0]
                lease.release.assert_not_called()
                self.assertGreaterEqual(lease._fd, 0, "the recoverable guard was closed")
                os.fstat(lease._fd)
                self.assertIs(daemon._failed_dictation_cleanup[(turn.id, "provider")], engine)
                self.assertEqual(len(sent), 1)
                self.assertEqual(sent[0]["code"], "unavailable")
                # The actual lifecycle retry, not a test-only acknowledgement.
                daemon._retry_dictation_cleanup(1)
                self.assertEqual(daemon._failed_dictation_cleanup, {})
                self.assertIsNone(engine._process)
                self.assertIsNotNone(child.returncode)
                lease.release.assert_called_once_with(cleanup_complete=True)
                self.assertEqual(lease._fd, -1)
                with self.execution(job_id="healthy-next", deadline_monotonic=time.monotonic() + .2) as next_grant:
                    next_grant.reaped()

    def test_shutdown_keeps_unproven_grant_and_reports_exit_quarantine(self):
        daemon = _worker_daemon([], jobs.JobLedger())
        daemon._cfg = {}
        daemon._lock = threading.RLock()
        daemon._stopping = threading.Event()
        daemon._speech = daemon._dictation = daemon._player = daemon._socket = None
        daemon._socket_ino = None
        daemon._wake_w = -1
        daemon._arbiter = mock.Mock()
        daemon._warn = mock.Mock()
        turn = voiced._DictationTurn("unproven-grant", mock.Mock())
        engine = mock.Mock()
        engine.close.side_effect = InterruptedError("unproven")
        with self.execution(job_id=turn.id) as grant:
            self.assertIsNotNone(daemon._close_dictation_resources(turn, None, engine, grant))
        lease = grant._lease
        with mock.patch.object(lease, "release", wraps=lease.release) as releasing:
            daemon._shutdown()
            releasing.assert_not_called()
        self.assertIs(daemon._failed_dictation_cleanup[(turn.id, "provider")], engine)
        warnings = [c.args[0] for c in daemon._warn.call_args_list]
        self.assertTrue(any("shutdown" in w and "accelerator" in w and "quarantine" in w
                            for w in warnings), warnings)
        # Fixture containment closes without proof, as process exit does.
        # The component must refuse the next grant after the FD is closed.
        lease.release(cleanup_complete=False)
        with self.assertRaises(accelerator.AcceleratorRefused) as refused:
            with self.execution(job_id="after-exit", deadline_monotonic=time.monotonic() + .2):
                self.fail("unproven exit granted the accelerator")
        self.assertEqual(refused.exception.code, "unavailable")
        self.assertIn("quarantined", str(refused.exception))
