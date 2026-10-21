"""Cleanup boundaries with real synthetic recorders, no audio device."""
import contextlib
import io
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from tests.test_stt_tool import tool
from tests.test_terminal_outcomes import voiced, _worker_daemon
from voicelib import audio, jobs, processes, stt


class ObservedLock:
    """A real lock with an event at a selected caller's acquire boundary."""

    def __init__(self):
        self.lock = threading.Lock()
        self.observed_thread = None
        self.attempted = threading.Event()
        self.boundary = threading.Event()

    def __enter__(self):
        if threading.current_thread() is self.observed_thread:
            self.attempted.set()
            self.boundary.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux waitid and groups")
class CleanupTests(unittest.TestCase):
    def setUp(self):
        # Exercise the legacy kernel without pidfd_open on current kernels too.
        # The matrix and test_process_identity exercise the pidfd backend.
        backend = mock.patch.object(processes, '_pidfd_available', False)
        backend.start()
        self.addCleanup(backend.stop)

    def capture(self, command="import time; time.sleep(60)"):
        capture = audio.MicCapture({"audio": {"capture_cmd": [
            sys.executable, "-c", command]}})
        capture.start()
        process, reader = capture._process, capture._reader

        def cleanup():
            # Runs only after assertions; contain broken implementations too.
            try:
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                waitable = True
            except ChildProcessError:
                waitable = False
            if process.returncode is None and waitable:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # A mutant may already have reaped the leader. Containment must
            # not depend on that broken observer, or signal its released ID.
            with contextlib.suppress(ChildProcessError):
                subprocess.Popen.wait(process, timeout=3)
            reader.join(3)
            if process.stdout:
                process.stdout.close()
        self.addCleanup(cleanup)
        return capture, process

    def ui(self, capture):
        mic = tool.MicTest({}, 30)
        mic._capture = capture
        mic.poll = lambda: None
        ui = object.__new__(tool.Ui)
        ui._mic = mic
        ui._screen = mock.Mock()
        ui._draw = lambda: None
        ui._handle = lambda key: 0
        return ui, mic

    def second_signal(self, number, *, initial_signal=True):
        capture, process = self.capture()
        ui, mic = self.ui(capture)
        previous = {n: signal.SIG_DFL for n in (signal.SIGHUP, signal.SIGTERM)}
        for n in previous:
            old = signal.signal(n, signal.SIG_DFL)
            self.addCleanup(signal.signal, n, old)
        ownership = []
        original = audio.kill_group

        def boundary(child):
            ownership.append((mic._capture is capture, capture._process is child))
            # The second real signal arrives inside stop(), before killpg.
            signal.raise_signal(number)
            signal.raise_signal(number)
            original(child)

        ui._screen.getch.side_effect = (
            (lambda: signal.raise_signal(number)) if initial_signal else lambda: ord("q"))
        with mock.patch.object(audio, "kill_group", side_effect=boundary), \
                mock.patch.object(processes.os, "killpg", wraps=os.killpg) as group_signal:
            interrupted = False
            try:
                ui.run()
            except KeyboardInterrupt:
                interrupted = True
            self.assertGreaterEqual(group_signal.call_count, 1,
                                    "second signal abandoned capture before group termination")
            self.assertEqual(ownership, [(True, True)],
                             "cleanup released ownership before the group signal")
            self.assertIsNotNone(process.returncode, "recorder was not waited for")
            self.assertFalse(Path(f"/proc/{process.pid}").exists(), "recorder survived cleanup")
            self.assertIsNone(mic._capture)
            self.assertIsNone(capture._process)
            self.assertEqual(interrupted, initial_signal)
        self.assertEqual({n: signal.getsignal(n) for n in previous}, previous)

    def test_second_hangup_during_meter_cleanup(self):
        self.second_signal(signal.SIGHUP)

    def test_second_termination_during_meter_cleanup(self):
        self.second_signal(signal.SIGTERM)

    def test_second_ctrl_c_during_meter_cleanup(self):
        self.second_signal(signal.SIGINT)

    def test_mixed_meter_signals_through_signal_and_wait(self):
        numbers = (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)
        for initial in numbers:
            with self.subTest(initial=initial):
                capture, process = self.capture()
                ui, mic = self.ui(capture)
                previous = {n: signal.getsignal(n) for n in numbers}
                original_signal, original_wait = audio.kill_group, process.wait

                def interrupt():
                    signal.raise_signal(initial)

                def boundary(child):
                    for number in numbers * 2:
                        signal.raise_signal(number)
                    original_signal(child)

                def waiting(*args, **kwargs):
                    for number in tuple(reversed(numbers)) * 2:
                        signal.raise_signal(number)
                    return original_wait(*args, **kwargs)

                ui._screen.getch.side_effect = interrupt
                with mock.patch.object(audio, "kill_group", side_effect=boundary), \
                        mock.patch.object(process, "wait", side_effect=waiting), \
                        mock.patch.object(processes.os, "killpg", wraps=os.killpg) as send:
                    try:
                        ui.run()
                    except KeyboardInterrupt:
                        pass
                    self.assertEqual(send.call_count, 1, "mixed signals abandoned group cleanup")
                    self.assertIsNotNone(process.returncode, "mixed signals skipped the wait")
                self.assertIsNone(mic._capture)
                self.assertFalse(Path(f"/proc/{process.pid}").exists())
                self.assertEqual({n: signal.getsignal(n) for n in numbers}, previous)

    def test_main_reports_incomplete_meter_cleanup(self):
        capture, process = self.capture()
        ui, mic = self.ui(capture)
        ui._screen.getch.return_value = ord("q")
        previous = {n: signal.getsignal(n) for n in
                    (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)}
        stderr = io.StringIO()
        with mock.patch.object(tool.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(tool.sys.stdout, "isatty", return_value=True), \
                mock.patch.object(tool.curses, "wrapper", side_effect=lambda *_: ui.run()), \
                mock.patch.object(audio, "kill_group", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stderr(stderr):
            status = tool.main([])
        self.assertEqual(status, 130, "main lost the interactive interrupt status")
        self.assertTrue(stderr.getvalue().strip(), "interruption was not reported")
        self.assertIs(mic._capture, capture)
        self.assertIs(capture._process, process)
        self.assertEqual({n: signal.getsignal(n) for n in previous}, previous)
        mic.stop()
        self.assertFalse(Path(f"/proc/{process.pid}").exists())

    def test_main_returns_130_after_interactive_interrupt(self):
        capture, process = self.capture()
        ui, mic = self.ui(capture)
        ui._screen.getch.side_effect = lambda: signal.raise_signal(signal.SIGINT)
        previous = {n: signal.getsignal(n) for n in
                    (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)}
        with mock.patch.object(tool.sys.stdin, "isatty", return_value=True), \
                mock.patch.object(tool.sys.stdout, "isatty", return_value=True), \
                mock.patch.object(tool.curses, "wrapper", side_effect=lambda *_: ui.run()), \
                contextlib.redirect_stderr(io.StringIO()):
            status = tool.main([])
        self.assertEqual(status, 130)
        self.assertIsNone(mic._capture)
        self.assertIsNone(capture._process)
        self.assertIsNotNone(process.returncode)
        self.assertFalse(Path(f"/proc/{process.pid}").exists())
        self.assertEqual({n: signal.getsignal(n) for n in previous}, previous)

    def test_interrupt_during_restoration_restores_every_handler(self):
        numbers = (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)
        for boundary in numbers:
            with self.subTest(boundary=boundary):
                capture, process = self.capture()
                ui, mic = self.ui(capture)
                ui._screen.getch.return_value = ord("q")
                previous = {n: signal.getsignal(n) for n in numbers}
                old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
                original = signal.signal
                injected = []

                def setting(number, handler):
                    result = original(number, handler)
                    if number == boundary and handler is previous[number] and not injected:
                        injected.append(True)
                        signal.raise_signal(signal.SIGINT)
                    return result

                interrupted = False
                try:
                    with mock.patch.object(signal, "signal", side_effect=setting):
                        try:
                            ui.run()
                        except KeyboardInterrupt:
                            interrupted = True
                    self.assertTrue(injected, "restoration boundary was not reached")
                    self.assertEqual({n: signal.getsignal(n) for n in numbers}, previous,
                                     "interrupt abandoned the remaining dispositions")
                    self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, []), old_mask)
                    self.assertTrue(interrupted, "new interrupt was swallowed")
                    self.assertIsNone(mic._capture)
                    self.assertIsNone(capture._process)
                    self.assertIsNotNone(process.returncode)
                    self.assertFalse(Path(f"/proc/{process.pid}").exists())
                finally:
                    for number, handler in previous.items():
                        original(number, handler)
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    def test_normal_meter_exit_survives_signals_during_cleanup(self):
        for number in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
            with self.subTest(signal=number):
                self.second_signal(number, initial_signal=False)

    def test_interrupted_cleanup_retains_ownership_for_retry(self):
        capture, process = self.capture()
        _ui, mic = self.ui(capture)
        for operation in ("signal", "wait"):
            with self.subTest(operation=operation):
                target, name = ((audio, "kill_group") if operation == "signal"
                                else (process, "wait"))
                with mock.patch.object(target, name, side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        mic.stop()
                self.assertIs(mic._capture, capture, "meter lost its capture")
                self.assertIs(capture._process, process, "capture lost its recorder")
        mic.stop()
        self.assertFalse(Path(f"/proc/{process.pid}").exists())

    def ignored_signal(self, number):
        capture, process = self.capture()
        ui, _mic = self.ui(capture)
        previous = signal.signal(number, signal.SIG_IGN)
        self.addCleanup(signal.signal, number, previous)
        observed = []
        original = audio.kill_group

        def boundary(child):
            observed.append(signal.getsignal(number))
            original(child)

        def loop():
            observed.append(signal.getsignal(number))
            return ord("q")

        ui._screen.getch.side_effect = loop
        with mock.patch.object(audio, "kill_group", side_effect=boundary):
            ui.run()
        self.assertEqual(observed, [signal.SIG_IGN, signal.SIG_IGN])
        self.assertEqual(signal.getsignal(number), signal.SIG_IGN)
        self.assertFalse(Path(f"/proc/{process.pid}").exists())

    def test_meter_preserves_inherited_ignored_hangup(self):
        self.ignored_signal(signal.SIGHUP)

    def test_meter_preserves_inherited_ignored_termination(self):
        self.ignored_signal(signal.SIGTERM)

    def test_meter_preserves_inherited_ignored_interrupt(self):
        self.ignored_signal(signal.SIGINT)

    def test_meter_restores_custom_dispositions(self):
        def inherited(*_):
            self.fail("inherited handler called before cleanup completed")
        numbers = (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)
        for number in numbers:
            prior = signal.signal(number, inherited)
            self.addCleanup(signal.signal, number, prior)
        self.test_mixed_meter_signals_through_signal_and_wait()
        self.assertTrue(all(signal.getsignal(n) is inherited for n in numbers))

    def test_group_signal_failure_can_be_retried_before_sealing(self):
        process = SimpleNamespace(pid=424242, returncode=None)
        check = mock.patch.object(processes, '_waitid', return_value=None)
        check.start()
        self.addCleanup(check.stop)
        with mock.patch.object(processes.os, "killpg",
                               side_effect=[InterruptedError("before signal"), None]) as send:
            with self.assertRaises(InterruptedError):
                processes.kill_group(process)
            self.assertFalse(getattr(process, "_voice_group_signalled", False),
                             "failed signal sealed off the cleanup retry")
            processes.kill_group(process)
            processes.kill_group(process)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(send.call_args, mock.call(process.pid, signal.SIGKILL))

    def test_overlapping_group_callers_serialize_the_final_signal(self):
        process = SimpleNamespace(pid=424242, returncode=None)
        check = mock.patch.object(processes, '_waitid', return_value=None)
        check.start()
        self.addCleanup(check.stop)
        entered, release, second_started = (threading.Event() for _ in range(3))
        calls, errors = [], []

        def sending(*args):
            calls.append(args)
            entered.set()
            release.wait(2)

        def stop(second=False):
            if second:
                second_started.set()
            try:
                processes.kill_group(process)
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=stop)
        second = threading.Thread(target=stop, args=(True,))
        with mock.patch.object(processes.os, "killpg", side_effect=sending):
            try:
                first.start()
                self.assertTrue(entered.wait(1), "first caller never entered signal")
                second.start()
                self.assertTrue(second_started.wait(1))
                # Let the second caller overlap the deliberately blocked signal.
                second.join(.1)
            finally:
                release.set()
                first.join(2)
                if second.ident is not None:
                    second.join(2)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(calls, [(process.pid, signal.SIGKILL)],
                         "overlapping callers sent multiple final group signals")

    def test_group_owner_rejects_ignored_and_custom_sigchld(self):
        for disposition in (signal.SIG_IGN, lambda *_: None):
            with self.subTest(disposition=disposition), \
                    mock.patch.object(signal, "getsignal", return_value=disposition):
                with self.assertRaisesRegex(OSError, "default SIGCHLD"):
                    processes.require_waitable_children()

    def provider(self, engine_type):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   start_new_session=True)
        def cleanup():
            if process.returncode is None:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=3)
            process.stdin.close()
            process.stdout.close()
        self.addCleanup(cleanup)
        engine = object.__new__(engine_type)
        engine._process = process
        engine._lifecycle = threading.Lock()
        engine._closed = engine._busy = False
        engine._ready = True
        engine._rate = 16000
        engine._links = None
        engine._held = {}
        engine._release = mock.Mock()
        engine._require_unchanged = mock.Mock()
        return engine, process

    def test_provider_teardown_failure_retains_process_for_retry(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            for method in ("_stop_child", "close"):
                for operation in ("signal", "wait"):
                    with self.subTest(engine=engine_type.__name__, method=method,
                                      operation=operation):
                        engine, process = self.provider(engine_type)
                        target, name = ((processes.os, "killpg") if operation == "signal"
                                        else (process, "wait"))
                        error = (InterruptedError("injected before signal") if operation == "signal"
                                 else subprocess.TimeoutExpired(process.args, 5))
                        caught = None
                        with mock.patch.object(target, name, side_effect=error):
                            try:
                                getattr(engine, method)()
                            except (InterruptedError, subprocess.TimeoutExpired) as failure:
                                caught = failure
                        self.assertIs(engine._process, process,
                                      "provider lost its retry handle before signal and reap completed")
                        self.assertIs(caught, error, "teardown failure was hidden")
                        with mock.patch.object(processes.os, "killpg", wraps=os.killpg) as send:
                            getattr(engine, method)()
                        self.assertEqual(send.call_count, int(operation == "signal"))
                        self.assertIsNotNone(process.returncode, "retry did not reap the provider")
                        self.assertFalse(Path(f"/proc/{process.pid}").exists())
                        self.assertIsNone(engine._process)

    def test_worker_preserves_stop_cause_after_provider_cleanup_failure(self):
        # Real _run_dictation -> _dictate -> _record_turn -> _record, with
        # real recorder/provider children. Only model construction, consent,
        # and the accelerator service are replaced; no audio device is used.
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            for cause in ("cancelled", "deadline"):
                for boundary in ("close", "reap"):
                    with self.subTest(engine=engine_type.__name__, cause=cause,
                                      boundary=boundary):
                        engine, process = self.provider(engine_type)
                        sent, ledger = [], jobs.JobLedger()
                        daemon = _worker_daemon(sent, ledger)
                        daemon._cfg = {"audio": {"capture_cmd": [
                            sys.executable, "-c", "import time; time.sleep(60)"]},
                            "stt": {"engine": "null", "max_seconds": 30}}
                        daemon._lock = threading.RLock()
                        daemon._arbiter = mock.Mock()
                        daemon._stopping = threading.Event()
                        daemon._require_capture_consent = lambda _: None
                        daemon._clear_dictation = lambda t: voiced.Daemon._clear_dictation(daemon, t)
                        daemon._warn = mock.Mock()
                        now = [0.0]
                        turn = voiced._DictationTurn("cleanup-" + cause, mock.Mock(),
                                                    deadline=1, clock=lambda: now[0])
                        turn.on_settle = ledger.record
                        daemon._dictation = turn
                        capture = audio.MicCapture(daemon._cfg)
                        self.addCleanup(capture.stop)
                        children = []
                        accepted = []

                        def reading(**kwargs):
                            children.append(capture._process)
                            if cause == "cancelled":
                                accepted.append(daemon._op_stop_dictation(
                                    {"id": "stop", "mode": "abort"})["stopped"])
                            # Expire even after abort: its earlier cause must win.
                            now[0] = 2.0
                            return None

                        grant = mock.Mock()
                        target, method = ((engine, "close") if boundary == "close"
                                          else (process, "wait"))
                        with mock.patch.object(voiced.audio, "MicCapture", return_value=capture), \
                                mock.patch.object(voiced.stt_lib, "make_stt", return_value=engine), \
                                mock.patch.object(voiced.accelerator, "execution",
                                                  return_value=contextlib.nullcontext(grant)), \
                                mock.patch.object(capture, "read", side_effect=reading), \
                                mock.patch.object(target, method, side_effect=
                                                  subprocess.TimeoutExpired(process.args, 5)):
                            daemon._run_dictation(turn)
                        row = ledger.get(turn.id)
                        with self.subTest(check="outcome"):
                            self.assertEqual((row["outcome"], row["code"]), (cause, cause))
                            self.assertEqual(len(sent), 1)
                            self.assertEqual(sent[0]["code"], cause)
                            self.assertNotIn("final", sent[0])
                        with self.subTest(check="diagnostic"):
                            messages = [str(call.args[0]) for call in daemon._warn.call_args_list]
                            self.assertTrue(any("cleanup failed" in m and "provider" in m
                                                and "TimeoutExpired" in m and turn.id in m
                                                for m in messages), messages)
                        with self.subTest(check="ownership"):
                            retained = getattr(daemon, "_failed_dictation_cleanup", None) or {}
                            self.assertIs(retained.get((turn.id, "provider")), engine,
                                          "settled worker discarded the uncompleted provider")
                        self.assertEqual(accepted, [True] if cause == "cancelled" else [])
                        self.assertIsNone(daemon._dictation)
                        daemon._arbiter.end_listen.assert_called_once_with(turn.id)
                        turn.receiver.close.assert_called_once()
                        grant.reaped.assert_not_called()
                        self.assertTrue(children, "test never reached recording")
                        self.assertIsNone(capture._process)
                        self.assertIsNotNone(children[0].returncode)
                        self.assertFalse(Path(f"/proc/{children[0].pid}").exists())
                        self.assertIs(engine._process, process)
                        retry = getattr(daemon, "_close_dictation_resources", None)
                        self.assertTrue(callable(retry), "daemon cannot retry its retained owners")
                        self.assertIsNone(retry(turn, capture, engine))
                        self.assertEqual(daemon._failed_dictation_cleanup, {})
                        self.assertIsNone(engine._process)
                        self.assertFalse(Path(f"/proc/{process.pid}").exists())

    def test_worker_attempts_both_cleanups_and_keeps_each_failed_owner(self):
        for provider_fails in (False, True):
            with self.subTest(provider_fails=provider_fails):
                sent, ledger = [], jobs.JobLedger()
                daemon = _worker_daemon(sent, ledger)
                daemon._cfg = {}
                daemon._warn = mock.Mock()
                capture, engine, grant = mock.Mock(), mock.Mock(), mock.Mock()
                capture.stop.side_effect = subprocess.TimeoutExpired("recorder", 1)
                if provider_fails:
                    engine.close.side_effect = subprocess.TimeoutExpired("provider", 5)
                turn = voiced._DictationTurn("two-cleanups", mock.Mock())
                turn.on_settle = ledger.record

                def record(*_):
                    turn.abort = True
                    turn.stop.set()
                    turn.stop.check()

                daemon._record = record
                daemon._dictate = lambda t: daemon._record_turn(t, SimpleNamespace(), grant)
                with mock.patch.object(voiced.audio, "MicCapture", return_value=capture), \
                        mock.patch.object(voiced.stt_lib, "make_stt", return_value=engine):
                    daemon._run_dictation(turn)
                self.assertEqual(ledger.get(turn.id)["outcome"], "cancelled")
                self.assertEqual(capture.stop.call_count, 2)  # initial + settlement retry
                self.assertEqual(engine.close.call_count, 2 if provider_fails else 1)
                if provider_fails:
                    grant.reaped.assert_not_called()
                else:
                    grant.reaped.assert_called_once()
                expected = {(turn.id, "capture"): capture}
                if provider_fails:
                    expected[(turn.id, "provider")] = engine
                self.assertEqual(daemon._failed_dictation_cleanup, expected)
                self.assertEqual(daemon._warn.call_count, 2 * len(expected))
                capture.stop.side_effect = engine.close.side_effect = None
                self.assertIsNone(daemon._close_dictation_resources(turn, capture, engine))
                self.assertEqual(daemon._failed_dictation_cleanup, {})

    def test_primary_decoder_error_survives_provider_reap_failure(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            with self.subTest(provider=engine_type.__name__):
                engine, process = self.provider(engine_type)
                sent, ledger = [], jobs.JobLedger()
                daemon = _worker_daemon(sent, ledger)
                daemon._cfg = {}
                daemon._warn = mock.Mock()
                capture = mock.Mock()
                turn = voiced._DictationTurn("decoder-error", mock.Mock())
                turn.on_settle = ledger.record
                daemon._record = mock.Mock(side_effect=stt.SttError("primary decoder error"))
                daemon._dictate = lambda t: daemon._record_turn(t, SimpleNamespace(), None)
                with mock.patch.object(voiced.audio, "MicCapture", return_value=capture), \
                        mock.patch.object(voiced.stt_lib, "make_stt", return_value=engine), \
                        mock.patch.object(process, "wait", side_effect=
                                          subprocess.TimeoutExpired("private args", 5)):
                    daemon._run_dictation(turn)
                self.assertEqual(ledger.get(turn.id)["code"], "unavailable")
                self.assertEqual(len(sent), 1)
                self.assertEqual(sent[0]["code"], "unavailable")
                self.assertIn("primary decoder error", sent[0]["error"])
                self.assertIs(daemon._failed_dictation_cleanup[(turn.id, "provider")], engine)
                self.assertIs(engine._process, process)
                messages = [c.args[0] for c in daemon._warn.call_args_list]
                self.assertTrue(any(turn.id in m and "provider" in m and "TimeoutExpired" in m
                                    for m in messages), messages)
                self.assertNotIn("private args", str(messages))
                self.assertIsNone(daemon._close_dictation_resources(turn, None, engine))
                self.assertEqual(daemon._failed_dictation_cleanup, {})
                self.assertIsNone(engine._process)

    def test_provider_teardown_methods_contend_on_lifecycle_lock(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            for method in ("_stop_child", "close", "cancel"):
                with self.subTest(engine=engine_type.__name__, method=method):
                    engine, process = self.provider(engine_type)
                    lock = engine._lifecycle = ObservedLock()
                    done, signalled = threading.Event(), threading.Event()
                    errors = []
                    original = stt.kill_group

                    def sending(child):
                        signalled.set()
                        lock.boundary.set()
                        original(child)

                    def stopping():
                        try:
                            getattr(engine, method)()
                        except BaseException as error:
                            errors.append(error)
                        finally:
                            done.set()
                            lock.boundary.set()

                    worker = lock.observed_thread = threading.Thread(target=stopping)
                    with mock.patch.object(stt, "kill_group", side_effect=sending):
                        try:
                            with lock:
                                worker.start()
                                reached = lock.boundary.wait(2)
                                attempted = lock.attempted.is_set()
                                premature = done.is_set() or signalled.is_set()
                        finally:
                            worker.join(3)
                    self.assertTrue(reached, "worker did not reach a lifecycle boundary")
                    self.assertTrue(attempted, "teardown bypassed the held lifecycle lock")
                    self.assertFalse(premature, "teardown ran while another owner held the lock")
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(errors, [])
                    engine.close()
                    self.assertIsNone(engine._process)
                    self.assertIsNotNone(process.returncode)

    def test_close_waits_for_inflight_provider_startup(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            with self.subTest(engine=engine_type.__name__), tempfile.TemporaryDirectory() as root:
                engine, _ = self.provider(engine_type)
                engine.close()
                engine._closed = False
                engine._resource_lifecycle = processes.Resource.lifecycle.fget(object.__new__(engine_type))
                lock = engine._lifecycle = ObservedLock()
                engine._command = lambda: [sys.executable, "-c", "import time; time.sleep(60)"]
                entered, release, closed = (threading.Event() for _ in range(3))
                children, errors = [], []
                original = processes.GroupPopen

                def launch(*args, **kwargs):
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("test never released process creation")
                    child = original(*args, **kwargs)
                    children.append(child)
                    return child

                def starting():
                    try:
                        engine._start()
                    except BaseException as error:
                        errors.append(error)

                def closing():
                    try:
                        engine.close()
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        closed.set()
                        lock.boundary.set()

                startup = threading.Thread(target=starting)
                closer = lock.observed_thread = threading.Thread(target=closing)
                try:
                    with mock.patch.object(stt.paths, "session_dir", return_value=root), \
                            mock.patch.object(stt, "GroupPopen", side_effect=launch):
                        try:
                            startup.start()
                            self.assertTrue(entered.wait(2))
                            closer.start()
                            self.assertTrue(lock.boundary.wait(2))
                            contended = lock.attempted.is_set()
                            premature = engine.cleanup_complete
                        finally:
                            release.set()
                            startup.join(3)
                            if closer.ident is not None:
                                closer.join(3)
                    self.assertFalse(startup.is_alive() or closer.is_alive())
                    self.assertTrue(all(isinstance(e, stt.SttError) for e in errors), errors)
                    self.assertTrue(closed.is_set(), "close did not respect its bound")
                    self.assertFalse(premature, "close returned while launch was in flight")
                    self.assertEqual(len(children), 1)
                    self.assertIsNone(engine._process)
                    self.assertIsNotNone(children[0].returncode)
                    self.assertFalse(Path(f"/proc/{children[0].pid}").exists())
                finally:
                    for child in children:
                        if child.returncode is None:
                            child.kill()
                        child.wait(timeout=3)
                        child.stdin.close()
                        child.stdout.close()
                    engine.close()

    def test_busy_failed_close_hands_group_signal_to_decode_finalizer(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            with self.subTest(engine=engine_type.__name__):
                engine, process = self.provider(engine_type)
                entered, release = threading.Event(), threading.Event()
                errors, sealed_before_wait = [], []
                original_reap = engine._reap

                def writing(*_):
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("test never released decode")

                def reaping(child):
                    sealed = getattr(child, "_voice_group_signalled", False)
                    sealed_before_wait.append(sealed)
                    if not sealed:
                        # Contain a broken finalizer by our fixture PID, then
                        # fail the assertion below instead of a wait timeout.
                        os.kill(child.pid, signal.SIGKILL)
                    original_reap(child)

                def decode():
                    try:
                        engine._decode(b"\0\0")
                    except BaseException as error:
                        errors.append(error)

                worker = threading.Thread(target=decode)
                with mock.patch.object(engine, "_write_all", side_effect=writing), \
                        mock.patch.object(engine, "_read_reply", return_value={"text": "words"}), \
                        mock.patch.object(engine, "_reap", side_effect=reaping), \
                        mock.patch.object(processes.os, "killpg", wraps=os.killpg) as send:
                    try:
                        worker.start()
                        self.assertTrue(entered.wait(2))
                        with mock.patch.object(stt, "kill_group", side_effect=InterruptedError):
                            with self.assertRaises(InterruptedError):
                                engine.close()
                        self.assertIs(engine._process, process)
                        self.assertTrue(engine._busy)
                        self.assertIsNone(processes.poll_exit(process))
                    finally:
                        release.set()
                        worker.join(3)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(sealed_before_wait, [True],
                                     "decode finalizer reaped before completing the failed group signal")
                    send.assert_called_once_with(process.pid, signal.SIGKILL)
                self.assertIsNone(engine._process)
                self.assertIsNotNone(process.returncode)
                self.assertFalse(Path(f"/proc/{process.pid}").exists())
                engine.close()

    def test_provider_reap_requests_a_finite_timeout(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            with self.subTest(engine=engine_type.__name__):
                process = SimpleNamespace(stdin=mock.Mock(), stdout=mock.Mock(), wait=mock.Mock())
                engine_type._reap(process)
                process.wait.assert_called_once()
                timeout = process.wait.call_args.kwargs.get("timeout")
                self.assertIsInstance(timeout, (int, float), "reap requested an unbounded wait")
                self.assertTrue(math.isfinite(timeout))
                self.assertGreater(timeout, 0)
                self.assertLessEqual(timeout, 5)

    def test_busy_provider_close_keeps_process_until_decode_reaps(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            with self.subTest(engine=engine_type.__name__):
                engine, process = self.provider(engine_type)
                entered, release = threading.Event(), threading.Event()
                errors = []

                def writing(*_):
                    entered.set()
                    release.wait(2)

                def decode():
                    try:
                        engine._decode(b"\0\0")
                    except BaseException as error:
                        errors.append(error)

                worker = threading.Thread(target=decode)
                with mock.patch.object(engine, "_write_all", side_effect=writing), \
                        mock.patch.object(engine, "_read_reply", return_value={"text": "words"}):
                    try:
                        worker.start()
                        self.assertTrue(entered.wait(1))
                        engine.close()
                        self.assertIs(engine._process, process,
                                      "busy close lost ownership before the decode worker reaped")
                        self.assertIsNone(process.returncode)
                    finally:
                        release.set()
                        worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertIsNone(engine._process)
                self.assertIsNotNone(process.returncode)
                self.assertFalse(Path(f"/proc/{process.pid}").exists())

    def test_receiver_loss_stops_capture_before_blocked_provider_cleanup(self):
        capture, process = self.capture()
        entered, release = threading.Event(), threading.Event()
        turn = voiced._DictationTurn("blocked-cleanup", mock.Mock())
        turn.capture = capture

        def interrupt():
            entered.set()
            release.wait(3)
        turn.interrupt = interrupt
        done = threading.Event()
        turn.receiver_alive = lambda: False
        daemon = object.__new__(voiced.Daemon)
        worker = threading.Thread(target=daemon._watch_dictation_receiver,
                                  args=(turn, done))
        try:
            with mock.patch.object(capture, "stop", wraps=capture.stop) as stop:
                started = time.monotonic()
                worker.start()
                self.assertTrue(entered.wait(1), "loss detection never reached provider cleanup")
                deadline = started + 1
                while time.monotonic() < deadline:
                    try:
                        info = os.waitid(os.P_PID, process.pid,
                                         os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    except ChildProcessError:
                        self.assertIsNotNone(process.returncode)
                        break
                    if info is not None:
                        break
                    time.sleep(.005)
                else:
                    self.fail("capture remained live while provider cleanup was blocked")
                self.assertTrue(worker.is_alive(), "fixture did not block provider cleanup")
                self.assertFalse(release.is_set())
                self.assertTrue(turn.abort)
                self.assertTrue(turn.stop.is_set())
                stop.assert_not_called()
        finally:
            release.set()
            worker.join(3)
            capture.stop()

    def test_capture_reader_keeps_exited_leader_until_group_signal(self):
        capture, process = self.capture("pass")
        capture._reader.join(2)
        self.assertFalse(capture._reader.is_alive())
        self.assertIsNone(process.returncode, "capture reader reaped the group leader")
        self.assertTrue(Path(f"/proc/{process.pid}").exists(),
                        "capture reader released the group identity")
        self.assert_signal_before_reap(capture, process)

    def test_observe_signal_exit_without_reaping(self):
        child = subprocess.Popen([sys.executable, "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"], start_new_session=True)
        try:
            self.assertEqual(processes.wait_exit(child, 2), -signal.SIGTERM)
            self.assertEqual(processes.poll_exit(child), -signal.SIGTERM)
            self.assertIsNone(child.returncode, "observation reaped the group leader")
            self.assertTrue(Path(f"/proc/{child.pid}").exists())
            info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            self.assertIsNotNone(info, "leader was no longer waitable")
            self.assertEqual(info.si_status, signal.SIGTERM)
            processes.kill_group(child)
            self.assertEqual(child.wait(timeout=2), -signal.SIGTERM)
        finally:
            if child.returncode is None:
                child.kill()
                child.wait(timeout=3)

    def test_wait_observation_timeout_boundary(self):
        child = SimpleNamespace(args=["fixture"])
        # Equal to the deadline and just past it must both use the documented
        # exception, without handing a negative duration to sleep().
        for timeout, end in ((0, 10), (.01, 10.02)):
            with self.subTest(timeout=timeout):
                caught = None
                with mock.patch.object(processes, "poll_exit", return_value=None), \
                        mock.patch.object(processes.time, "monotonic", side_effect=[10, end, end + .01]):
                    try:
                        processes.wait_exit(child, timeout)
                    except Exception as error:
                        caught = error
                self.assertIsInstance(caught, subprocess.TimeoutExpired,
                                      f"wait expiry raised {type(caught).__name__}")
                self.assertEqual(caught.timeout, timeout)

    def test_capture_publishes_eof_when_recorder_keeps_running(self):
        capture = audio.MicCapture({"audio": {"capture_cmd": [sys.executable, "-c",
            "import os, time; os.close(1); time.sleep(60)"]}})
        capture._TERM_TIMEOUT = .03
        self.addCleanup(capture.stop)
        errors = []
        # Record a broken reader as evidence for assertions, rather than
        # letting an unhandled thread exception masquerade as a test catch.
        with mock.patch.object(threading, "excepthook", side_effect=lambda args: errors.append(args.exc_value)):
            capture.start()
            child, reader = capture._process, capture._reader
            reader.join(2)
        self.assertFalse(reader.is_alive(), "EOF reader did not finish its bounded observation")
        self.assertEqual(errors, [], "reader missed the timeout exception contract")
        try:
            item = capture._queue.get(timeout=.1)
        except queue.Empty:
            item = None
        self.assertIs(item, audio._EOF, "reader failed to publish EOF while recorder stayed alive")
        self.assertIsNone(processes.poll_exit(child))
        self.assertIsNone(child.returncode)
        self.assertIs(capture._process, child, "EOF reader discarded recorder ownership")
        self.assertTrue(Path(f"/proc/{child.pid}").exists())
        capture.stop()
        self.assertIsNotNone(child.returncode)
        self.assertFalse(Path(f"/proc/{child.pid}").exists())

    def test_group_helper_refuses_an_already_reaped_object(self):
        with subprocess.Popen([sys.executable, "-c", "pass"],
                              start_new_session=True) as process:
            process.wait(timeout=2)
            # Never send a real signal to this now-unowned numeric ID.
            with mock.patch.object(processes.os, "killpg") as send:
                processes.kill_group(process)
                processes.kill_group(process)
            send.assert_not_called()
            self.assertFalse(getattr(process, '_voice_outside_reap', False),
                             'a known exit was incorrectly reported as an outside reap')

    def test_capture_refuses_automatic_child_reaping_before_launch(self):
        capture = audio.MicCapture({"audio": {"capture_cmd": [sys.executable, "-c", "pass"]}})
        with mock.patch.object(signal, "getsignal", return_value=signal.SIG_IGN), \
                mock.patch.object(audio.subprocess, "Popen") as launch:
            # Mocking creation keeps even the broken implementation harmless.
            launch.side_effect = OSError("fixture launch refused")
            with self.assertRaises(audio.AudioError):
                capture.start()
            launch.assert_not_called()

    def test_providers_refuse_automatic_child_reaping_before_launch(self):
        for engine_type in (stt.WhisperStt, stt.WhistleStt):
            with self.subTest(engine=engine_type.__name__):
                engine = object.__new__(engine_type)
                engine._held = {}
                engine._closed = False
                engine._lifecycle = threading.Lock()
                engine._require_unchanged = lambda _: None
                engine._command = lambda: [sys.executable, "-c", "pass"]
                engine._links = None
                try:
                    with mock.patch.object(signal, "getsignal", return_value=signal.SIG_IGN), \
                            mock.patch.object(stt.subprocess, "Popen") as launch:
                        launch.side_effect = OSError("fixture launch refused")
                        with self.assertRaises(OSError):
                            engine._start()
                        launch.assert_not_called()
                finally:
                    if engine._links:
                        os.rmdir(engine._links)

    def test_concurrent_capture_stops_send_only_one_final_group_signal(self):
        capture, process = self.capture()
        original = os.killpg
        with mock.patch.object(processes.os, "killpg", wraps=original) as send:
            workers = [threading.Thread(target=capture.request_stop) for _ in range(8)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(2)
                self.assertFalse(worker.is_alive())
            error = None
            try:
                capture.stop()
            except OSError as caught:
                error = caught
            self.assertIsNone(error, 'concurrent stops did not seal before the final reap')
            processes.kill_group(process)
            send.assert_called_once_with(process.pid, signal.SIGKILL)

    def assert_signal_before_reap(self, owner, process):
        original = os.killpg
        observed = []

        def signal_group(pgid, number):
            # A waitable zombie proves this is still our child's identity.
            try:
                info = os.waitid(os.P_PID, process.pid,
                                 os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                info = None
            observed.append(info is not None and info.si_pid == process.pid)
            if observed[-1]:
                original(pgid, number)

        with mock.patch.object(processes.os, "killpg", side_effect=signal_group) as send:
            owner.stop()
            count = send.call_count
            processes.kill_group(process)
            processes.kill_group(process)
            self.assertEqual(send.call_count, count, "group signalled after its leader was reaped")
        self.assertEqual(observed, [True], "leader was not retained through final group signal")
        self.assertFalse(Path(f"/proc/{process.pid}").exists())

    def provider_exit(self, engine_type, reply=False):
        process = subprocess.Popen([sys.executable, "-c", "pass"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   start_new_session=True)
        def cleanup():
            if process.returncode is None:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=3)
            process.stdin.close()
            process.stdout.close()
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 2
        while os.waitid(os.P_PID, process.pid,
                        os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            self.assertLess(time.monotonic(), deadline, "fixture leader did not exit")
            time.sleep(.005)
        capture = audio.MicCapture()
        capture._process = process
        engine = object.__new__(engine_type)
        engine._closed = False
        engine._lifecycle = threading.Lock()
        engine._process = process
        engine._stdout = bytearray()
        with self.assertRaises(stt.SttError):
            if reply:
                engine._read_reply(process, time.monotonic() + 1, "loading")
            else:
                engine.check_health()
        self.assertIsNone(process.returncode, "provider observation reaped the group leader")
        self.assert_signal_before_reap(capture, process)

    def test_whisper_health_keeps_exited_leader(self):
        self.provider_exit(stt.WhisperStt)

    def test_whistle_health_keeps_exited_leader(self):
        self.provider_exit(stt.WhistleStt)

    def test_provider_eof_keeps_exited_leader(self):
        self.provider_exit(stt.WhisperStt, reply=True)


if __name__ == "__main__":
    unittest.main()
