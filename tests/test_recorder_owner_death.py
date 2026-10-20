"""Pin the documented SIGKILL limit using synthetic stall and PCM recorders."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

from tests import test_daemon, test_dictation_liveness as live


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux process ownership")
class RecorderOwnerDeathTests(unittest.TestCase):
    _stop_daemon = live.DictationLivenessTests._stop_daemon
    _log_tail = live.DictationLivenessTests._log_tail
    _wait_until_serving = live.DictationLivenessTests._wait_until_serving
    _send = live.DictationLivenessTests._send
    _environment = live.DictationLivenessTests._environment
    entries = live.DictationLivenessTests.entries
    wait_for = live.DictationLivenessTests.wait_for
    receiver = live.DictationLivenessTests.receiver
    start_turn = live.DictationLivenessTests.start_turn

    def setUp(self):
        # Adopt our orphan fixtures so even a non-reaping container PID 1
        # cannot accumulate zombies from the deliberately tested limitation.
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        self.addCleanup(libc.prctl, 36, previous.value, 0, 0, 0)
        self.meter = None
        test_daemon.DaemonTestCase.setUp(self)
        self.addCleanup(self.clean_recorders)

    def clean_recorders(self):
        if self.meter is not None:
            if self.meter.poll() is None:
                self.meter.kill()
            self.meter.wait(timeout=3)
        self._stop_daemon()
        for key in ("capture", "provider"):
            for row in self.entries(key):
                pid = row["pid"]
                try:
                    # These children are now ours, retained until this reap.
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 3
                while True:
                    try:
                        reaped, _ = os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if reaped:
                        break
                    self.assertLess(time.monotonic(), deadline, "fixture cleanup timed out")
                    time.sleep(.005)
                self.assertFalse(Path(f"/proc/{pid}").exists(), "fixture survived test cleanup")

    @staticmethod
    def state(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            return None

    def assert_owner_death_limit(self, pid, *, stalled):
        if stalled:
            # This is the documented limit, not an acoustic-silence fixture:
            # no write means no opportunity to observe the closed pipe.
            deadline = time.monotonic() + .5
            while time.monotonic() < deadline:
                self.assertNotIn(self.state(pid), (None, "Z"),
                                 "no-frame recorder owner-death behaviour changed")
                time.sleep(.01)
        else:
            self.wait_for(lambda: self.state(pid) in (None, "Z"), 1,
                          "writing PCM recorder did not discover its dead owner")
        print(f"OWNER-DEATH {self._testMethodName}: recorder={pid} "
              f"state={self.state(pid)} stalled={stalled}", flush=True)

    def daemon_death(self, *, stalled):
        receiver = self.receiver()
        _turn, pids = self.start_turn(receiver)
        self.daemon.kill()
        self.daemon.wait(timeout=3)
        self.assert_owner_death_limit(pids["capture"], stalled=stalled)

    def test_daemon_sigkill_without_frames(self):
        self.daemon_death(stalled=True)

    def test_daemon_sigkill_during_silent_pcm(self):
        self.daemon_death(stalled=False)

    def meter_death(self, *, stalled):
        root = Path(self.root)
        repo = Path(test_daemon.DAEMON).parent
        script = root / "meter.py"
        script.write_text(
            "import runpy, sys, time\n"
            "from unittest import mock\n"
            f"sys.path.insert(0, {str(repo)!r})\n"
            f"tool = runpy.run_path({str(repo / 'kilix-stt')!r})\n"
            f"mic = tool['MicTest']({{'audio': {{'capture_cmd': "
            f"[sys.executable, {str(root / 'capture.py')!r}]}}}}, 30)\n"
            "mic.start()\n"
            "ui = object.__new__(tool['Ui'])\n"
            "ui._mic = mic\n"
            "ui._draw = lambda: None\n"
            "ui._screen = mock.Mock()\n"
            "ui._screen.getch.side_effect = lambda: (time.sleep(.02), -1)[1]\n"
            "ui.run()\n")
        env = dict(os.environ, TEST_PROCESSES=str(self.process_log),
                   TEST_CAPTURE_FILE=str(self.capture_mode))
        self.meter = subprocess.Popen([sys.executable, str(script)], env=env,
                                      stdout=subprocess.DEVNULL, stderr=self.log)
        self.wait_for(lambda: bool(self.entries("capture")), 3, "meter recorder did not start")
        pid = self.entries("capture")[-1]["pid"]
        self.assertIsNone(self.meter.poll(), "meter exited before SIGKILL")
        self.meter.kill()
        self.meter.wait(timeout=3)
        self.assert_owner_death_limit(pid, stalled=stalled)

    def test_meter_sigkill_without_frames(self):
        self.meter_death(stalled=True)

    def test_meter_sigkill_during_silent_pcm(self):
        self.meter_death(stalled=False)


if __name__ == "__main__":
    unittest.main()
