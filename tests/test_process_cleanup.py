"""Cleanup boundaries with real synthetic recorders, no audio device."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from tests.test_stt_tool import tool
from tests.test_terminal_outcomes import voiced
from voicelib import audio, processes, stt


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux waitid and groups")
class CleanupTests(unittest.TestCase):
    def capture(self, command="import time; time.sleep(60)"):
        capture = audio.MicCapture({"audio": {"capture_cmd": [
            sys.executable, "-c", command]}})
        capture.start()
        process, reader = capture._process, capture._reader

        def cleanup():
            # Runs only after assertions; contain broken implementations too.
            if process.returncode is None:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=3)
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

    def test_normal_meter_exit_survives_signals_during_cleanup(self):
        for number in (signal.SIGHUP, signal.SIGTERM):
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

    def test_group_helper_refuses_an_already_reaped_object(self):
        with subprocess.Popen([sys.executable, "-c", "pass"],
                              start_new_session=True) as process:
            process.wait(timeout=2)
            # Never send a real signal to this now-unowned numeric ID.
            with mock.patch.object(processes.os, "killpg") as send:
                processes.kill_group(process)
                processes.kill_group(process)
            send.assert_not_called()

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
            capture.stop()
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
