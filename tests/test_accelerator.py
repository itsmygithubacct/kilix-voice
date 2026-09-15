"""S04 / LEASE-02: kilix-voice runs no accelerator engine without the one lease.

Every test here drives the real kilix_device_lease module from its one home,
kilix-system-monitor's kilix-device-lease component, and nothing is copied
into kilix-voice. Its source must be on the import path: ``make test`` and
``make test-clean`` put LEASE_SRC there. When it is not, these tests FAIL,
naming LEASE_SRC; they never skip, because an accelerator engine running
without the lease is exactly what they exist to rule out.

No test here may touch the namespace the providers on this machine share.
Every acquire is checked to name a private namespace under this test's own
temporary directory, and fails otherwise.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

try:
    import kilix_device_lease as LEASE
except ImportError:
    LEASE = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_accelerator", importlib.machinery.SourceFileLoader(
        "kilix_voiced_accelerator", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import accelerator, jobs, protocol, resources, stt as stt_lib  # noqa: E402

MISSING = ("kilix_device_lease is not importable. Run the suite with the "
           "kilix-device-lease source on PYTHONPATH: make test and make test-clean "
           "set it from LEASE_SRC (default ../../kilix-system-monitor/components/"
           "kilix-device-lease/src).")
_REAL_ACQUIRE = LEASE.acquire if LEASE is not None else None


def wait_until(predicate, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _token():
    """A job's real stop token, as the daemon hands one to the lease."""
    return voiced._DictationTurn("listen-1", mock.Mock()).stop


class _LeaseFixture(unittest.TestCase):

    def setUp(self) -> None:
        if LEASE is None:
            self.fail(MISSING)
        self.lease = LEASE
        self.parent = tempfile.mkdtemp(prefix="kv-lease-")
        self.addCleanup(shutil.rmtree, self.parent, True)
        self.namespace = os.path.join(self.parent, "leases")
        parent = self.parent

        def private_only(**kwargs):
            namespace = kwargs.get("namespace")
            if not isinstance(namespace, str) or not namespace.startswith(parent + os.sep):
                raise AssertionError(f"a test asked the shared lease namespace ({namespace!r})")
            return _REAL_ACQUIRE(**kwargs)

        patcher = mock.patch.object(LEASE, "acquire", private_only)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.token = _token()

    def execution(self, **over):
        kwargs = dict(device_class=resources.DEVICE_CUDA, task="transcribe",
                      job_id="listen-1", deadline_monotonic=time.monotonic() + 30,
                      token=self.token, device_label="gpu0", namespace=self.namespace)
        kwargs.update(over)
        return accelerator.execution(**kwargs)

    def peer(self) -> subprocess.Popen:
        """Another process, holding the accelerator until told to release it."""
        script = ("import sys, time, kilix_device_lease as lease\n"
                  "held = lease.acquire(job_id='peer-1', workload='stt-job', device='gpu0',\n"
                  "                     deadline=time.monotonic() + 60, namespace=sys.argv[1])\n"
                  "print('held', flush=True)\n"
                  "sys.stdin.readline()\n"
                  "held.release(cleanup_complete=True)\n"
                  "print('released', flush=True)\n")
        source = os.path.dirname(os.path.dirname(os.path.realpath(LEASE.__file__)))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=source)
        proc = subprocess.Popen([sys.executable, "-B", "-c", script, self.namespace],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                env=env)

        def stop() -> None:
            if proc.poll() is None:
                proc.kill()
            proc.wait(10)
            proc.stdin.close()
            proc.stdout.close()

        self.addCleanup(stop)
        self.assertEqual(proc.stdout.readline().strip(), "held")
        return proc

    def release(self, proc: subprocess.Popen) -> None:
        proc.stdin.write("\n")
        proc.stdin.flush()
        self.assertEqual(proc.stdout.readline().strip(), "released")
        proc.wait(10)


class LeaseHomeTestCase(unittest.TestCase):

    def test_the_lease_home_is_importable_and_speaks_this_clients_version(self) -> None:
        if LEASE is None:
            self.fail(MISSING)
        self.assertEqual(LEASE.VERSION, accelerator.LEASE_VERSION)
        self.assertTrue(set(accelerator.WORKLOAD_FOR_TASK.values()) <= set(LEASE.WORKLOADS))
        document = os.path.join(os.path.dirname(os.path.realpath(LEASE.__file__)), os.pardir,
                                os.pardir, os.pardir, os.pardir, "contracts",
                                "kilix.device-lease-v1.interface.json")
        if os.path.exists(document):
            with open(document, "rb") as handle:
                raw = handle.read()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), accelerator.LEASE_INTERFACE_SHA256,
                             "the lease interface changed since this client was written")
            self.assertEqual(json.loads(raw)["label_pattern"],
                             accelerator.DEVICE_LABEL.pattern)


class CpuEngineTestCase(_LeaseFixture):

    def test_cpu_engine_acquires_nothing(self) -> None:
        with mock.patch.object(accelerator, "_module",
                               side_effect=AssertionError("the lease was imported")):
            with self.execution(device_class=resources.DEVICE_CPU, device_label="") as grant:
                self.assertIsNone(grant)
        self.assertEqual(os.listdir(self.parent), [], "a CPU engine touched the namespace")

    def test_every_catalogued_recogniser_resolves_to_the_cpu(self) -> None:
        for model in stt_lib.models.MODEL_IDS:
            with self.subTest(model=model):
                resolved = stt_lib.resolve_stt({"stt": {"engine": "vosk", "model": model}})
                self.assertEqual((resolved.device_class, resolved.task),
                                 (resources.DEVICE_CPU, "transcribe"))


class GrantTestCase(_LeaseFixture):

    def test_accelerator_engine_waits_for_grant_before_construction(self) -> None:
        peer = self.peer()
        built = []

        def engine_run() -> None:
            with self.execution() as grant:
                built.append((time.monotonic(), grant.guard_fd >= 0))
                grant.reaped()

        worker = threading.Thread(target=engine_run, daemon=True)
        worker.start()
        worker.join(0.6)
        self.assertEqual(built, [], "an engine was built while another process held the grant")
        released_at = time.monotonic()
        self.release(peer)
        worker.join(15)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(built), 1)
        self.assertGreaterEqual(built[0][0], released_at)
        self.assertIs(built[0][1], True)

    def test_cancel_and_deadline_while_queued_map_to_closed_codes(self) -> None:
        peer = self.peer()
        built = []
        stopper = threading.Timer(0.3, self.token.set)
        stopper.start()
        self.addCleanup(stopper.cancel)
        with self.assertRaises(accelerator.AcceleratorRefused) as caught:
            with self.execution():
                built.append("cancelled run")
        self.assertEqual(caught.exception.code, protocol.ERR_CANCELLED)
        with self.assertRaises(accelerator.AcceleratorRefused) as caught:
            with self.execution(token=_token(), deadline_monotonic=time.monotonic() + 0.3):
                built.append("expired run")
        self.assertEqual(caught.exception.code, protocol.ERR_DEADLINE)
        self.assertEqual(built, [])
        self.release(peer)

    def test_queue_full_is_busy(self) -> None:
        peer = self.peer()
        leave, queued, outcomes = threading.Event(), set(), []

        def waiter(index: int) -> None:
            try:
                granted = self.lease.acquire(
                    job_id=f"waiter-{index}", workload="stt-job", device="gpu0",
                    deadline=time.monotonic() + 60, cancelled=leave.is_set,
                    progress=lambda status, index=index: queued.add(index),
                    namespace=self.namespace)
            except self.lease.LeaseError as error:
                outcomes.append(error.code)
            else:
                granted.release(cleanup_complete=True)
                outcomes.append("granted")

        waiters = [threading.Thread(target=waiter, args=(i,), daemon=True)
                   for i in range(self.lease.MAX_WORKLOAD_QUEUE)]
        for thread in waiters:
            thread.start()
        try:
            self.assertTrue(wait_until(lambda: len(queued) == len(waiters), 20), queued)
            with self.assertRaises(accelerator.AcceleratorRefused) as caught:
                with self.execution(job_id="listen-9"):
                    self.fail("granted past a full queue")
            self.assertEqual(caught.exception.code, protocol.ERR_BUSY)
        finally:
            leave.set()
            for thread in waiters:
                thread.join(20)
        self.assertEqual(outcomes, ["cancelled"] * len(waiters))
        self.release(peer)

    def test_unproven_teardown_quarantines_next_grant(self) -> None:
        with self.execution() as grant:
            self.assertIsNotNone(grant)             # never marked reaped
        with self.assertRaises(accelerator.AcceleratorRefused) as caught:
            with self.execution(job_id="listen-2"):
                self.fail("granted after an unproven teardown")
        self.assertEqual(caught.exception.code, protocol.ERR_UNAVAILABLE)
        self.assertIn("quarantined", str(caught.exception))

    def test_an_engine_that_fails_still_releases_and_quarantines(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.execution():
                raise RuntimeError("the engine failed")
        with self.assertRaises(accelerator.AcceleratorRefused) as caught:
            with self.execution(job_id="listen-2", deadline_monotonic=time.monotonic() + 2):
                self.fail("granted after a failed engine")
        self.assertEqual(caught.exception.code, protocol.ERR_UNAVAILABLE)

    def test_a_reaped_teardown_frees_the_accelerator_for_the_next(self) -> None:   # control
        with self.execution() as grant:
            grant.reaped()
        with self.execution(job_id="listen-2") as again:
            self.assertIsNotNone(again)
            again.reaped()


class RefusalTestCase(_LeaseFixture):

    def test_a_missing_or_malformed_device_label_is_unavailable_before_the_lease(self) -> None:
        for label in ("", "has space", "-leading", "x" * 97, None):
            with self.subTest(label=label):
                with mock.patch.object(accelerator, "_module",
                                       side_effect=AssertionError("the lease was asked")):
                    with self.assertRaises(accelerator.AcceleratorRefused) as caught:
                        with self.execution(device_label=label):
                            self.fail("an engine ran without a device label")
                self.assertEqual(caught.exception.code, protocol.ERR_UNAVAILABLE)
                self.assertIn(accelerator.DEVICE_KEY, str(caught.exception))

    def test_every_lease_refusal_reaches_a_closed_code(self) -> None:
        table = (("queue-full", protocol.ERR_BUSY), ("cancelled", protocol.ERR_CANCELLED),
                 ("deadline", protocol.ERR_DEADLINE), ("unavailable", protocol.ERR_UNAVAILABLE),
                 ("lost-lease", protocol.ERR_UNAVAILABLE),
                 ("invalid-request", protocol.ERR_INTERNAL))
        self.assertEqual({code for code, _ in table}, set(json.loads(json.dumps(
            ["cancelled", "deadline", "invalid-request", "lost-lease", "queue-full",
             "unavailable"]))))
        for lease_code, code in table:
            with self.subTest(lease_code=lease_code):
                refusal = self.lease.LeaseError(lease_code, "refused")
                with mock.patch.object(self.lease, "acquire", side_effect=refusal):
                    with self.assertRaises(accelerator.AcceleratorRefused) as caught:
                        with self.execution():
                            self.fail("granted")
                self.assertEqual(caught.exception.code, code)

    def probe(self, path: list[str]) -> str:
        """Import voicelib.accelerator with exactly ``path`` first; how _module answers."""
        script = ("import sys\n"
                  "sys.path[:0] = sys.argv[1:]\n"
                  "from voicelib import accelerator\n"
                  "try:\n"
                  "    accelerator._module()\n"
                  "except accelerator.AcceleratorRefused as error:\n"
                  "    print(error.code)\n"
                  "else:\n"
                  "    print('accepted')\n")
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run([sys.executable, "-B", "-c", script, *path], cwd=self.parent,
                              env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.strip()

    def test_version_mismatch_and_local_copy_refused(self) -> None:
        source = os.path.dirname(os.path.dirname(os.path.realpath(LEASE.__file__)))
        other = os.path.join(self.parent, "other-version")
        os.makedirs(other)
        with open(os.path.join(other, "kilix_device_lease.py"), "w", encoding="utf-8") as handle:
            handle.write("VERSION = 'kilix.device-lease/v0'\n")
        copy = os.path.join(self.parent, "voice-copy")
        shutil.copytree(os.path.join(ROOT, "voicelib"), os.path.join(copy, "voicelib"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(os.path.realpath(LEASE.__file__),
                     os.path.join(copy, "voicelib", "kilix_device_lease.py"))
        cases = (
            ("the lease home", [source, ROOT], "accepted"),                  # control
            ("another version", [other, ROOT], protocol.ERR_UNAVAILABLE),
            ("a copy planted in voicelib", [os.path.join(copy, "voicelib"), copy],
             protocol.ERR_UNAVAILABLE),
            ("nothing installed", [ROOT], protocol.ERR_UNAVAILABLE),
        )
        for label, path, expected in cases:
            with self.subTest(case=label):
                self.assertEqual(self.probe(path), expected)


class DaemonTestCase(_LeaseFixture):
    """The real dictation worker, with an engine resolved to an accelerator."""

    def setUp(self) -> None:
        super().setUp()
        self.sent, self.ledger, self.built = [], jobs.JobLedger(), []
        d = object.__new__(voiced.Daemon)
        d._cfg = {"stt": {"engine": "vosk", "model_path": "/nonexistent", "max_seconds": 120,
                          "accelerator_device": "gpu0"}, "vad": {"silence_ms": 900}}
        d._stopping = threading.Event()
        d._warn = d._debug = lambda *a, **k: None
        d._send = lambda receiver, msg: self.sent.append(msg) or True
        d._clear_dictation = lambda turn: None
        d._touch = lambda: None
        d._require_capture_consent = lambda resolved=None: None
        d._jobs = self.ledger
        self.daemon = d
        self.device_class = resources.DEVICE_CUDA
        self.close_error: Exception | None = None
        frames = [b"\x00" * 320, b"\x00" * 320]
        capture = mock.Mock(rate=16000, frame_bytes=320, overruns=0, error="")
        capture.read.side_effect = lambda *a, **k: frames.pop(0) if frames else None
        events = iter([voiced.events.VAD_SPEECH_START, voiced.events.VAD_SPEECH_END])

        def resolve(cfg):
            return stt_lib.ResolvedStt(engine="vosk", model_id="accelerated", model_dir=None,
                                       settings_path=None, lib_path=None,
                                       device_class=self.device_class)

        def make_stt(*args, **kwargs):
            engine = mock.Mock(supports_partials=False)
            engine.feed.return_value = None
            engine.end_utterance.return_value = "the words"
            if self.close_error is not None:
                engine.close.side_effect = self.close_error
            self.built.append(time.monotonic())
            return engine

        for patcher in (
                mock.patch.object(voiced.stt_lib, "resolve_stt", resolve),
                mock.patch.object(voiced.stt_lib, "make_stt", make_stt),
                mock.patch.object(voiced.audio, "MicCapture", lambda cfg: capture),
                mock.patch.object(voiced, "Vad", lambda cfg: mock.Mock(
                    feed=lambda frame: next(events, ""))),
                mock.patch.object(voiced.accelerator, "execution", functools.partial(
                    accelerator.execution, namespace=self.namespace))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def dictate(self, turn_id: str = "listen-5", deadline: float | None = None):
        turn = voiced._DictationTurn(turn_id, mock.Mock(), deadline)
        turn.on_settle = self.ledger.record
        worker = threading.Thread(target=voiced.Daemon._run_dictation,
                                  args=(self.daemon, turn), daemon=True)
        worker.start()
        return turn, worker

    def next_grant(self) -> str:
        try:
            with self.execution(job_id="listen-next", deadline_monotonic=time.monotonic() + 2) as grant:
                grant.reaped()
        except accelerator.AcceleratorRefused as error:
            return error.code
        return "granted"

    def test_a_dictation_waits_for_the_grant_before_its_engine_is_built(self) -> None:
        peer = self.peer()
        _turn, worker = self.dictate()
        worker.join(0.6)
        self.assertEqual(self.built, [])
        released_at = time.monotonic()
        self.release(peer)
        worker.join(15)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(self.built), 1)
        self.assertGreaterEqual(self.built[0], released_at)
        self.assertEqual([m.get("final") for m in self.sent], ["the words"])
        self.assertEqual(self.next_grant(), "granted")     # closed cleanly, so released

    def test_a_turn_whose_budget_ends_in_the_queue_is_refused_deadline_and_builds_nothing(self) -> None:
        peer = self.peer()
        turn, worker = self.dictate(deadline=time.monotonic() + 0.3)
        worker.join(15)
        self.assertEqual(self.built, [])
        self.assertEqual([(m.get("code"), "final" in m) for m in self.sent],
                         [(protocol.ERR_DEADLINE, False)])
        self.assertEqual(self.ledger.get(turn.id)["outcome"], "deadline")
        self.release(peer)

    def test_no_device_label_is_unavailable_and_nothing_is_built(self) -> None:
        del self.daemon._cfg["stt"]["accelerator_device"]
        turn, worker = self.dictate()
        worker.join(15)
        self.assertEqual(self.built, [])
        self.assertEqual([m.get("code") for m in self.sent], [protocol.ERR_UNAVAILABLE])
        self.assertIn(accelerator.DEVICE_KEY, self.sent[0]["error"])
        self.assertEqual(os.listdir(self.parent), [])

    def test_an_engine_whose_close_fails_leaves_the_accelerator_quarantined(self) -> None:
        self.close_error = stt_lib.SttError("the recogniser would not close")
        _turn, worker = self.dictate()
        worker.join(15)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(self.next_grant(), protocol.ERR_UNAVAILABLE)

    def test_a_cpu_engine_dictates_without_the_lease(self) -> None:
        self.device_class = resources.DEVICE_CPU
        with mock.patch.object(accelerator, "_module",
                               side_effect=AssertionError("the lease was imported")):
            _turn, worker = self.dictate()
            worker.join(15)
        self.assertEqual([m.get("final") for m in self.sent], ["the words"])
        self.assertEqual(os.listdir(self.parent), [])


if __name__ == "__main__":
    unittest.main()
