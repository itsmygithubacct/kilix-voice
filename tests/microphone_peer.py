"""Private subprocess fixtures. Never opens a real audio device."""
import ctypes
import json
import os
from pathlib import Path
import signal
import resource
import subprocess
import sys
import tempfile
import time
lease_source = Path(os.environ.get('DEVICE_LEASE_TEST_SOURCE',
    str(Path(__file__).resolve().parents[2] / 'kilix-system-monitor')))
sys.path.insert(0, os.environ.get('DEVICE_LEASE_TEST_PATH',
    str(lease_source / 'components/kilix-device-lease/src')))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicelib import audio, device_leases
from voicelib.microphone import CaptureControl, CaptureOwner


def wait_for(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
    raise AssertionError("Fixture never started")


def unavailable(namespace):
    try:
        lease = device_leases.acquire(job_id="later", workload="stt-job", device="microphone",
                                      deadline=time.monotonic() + 0.2, namespace=namespace)
    except device_leases.LeaseError as error:
        return error.code == "unavailable"
    lease.release(cleanup_complete=True)
    return False


def recorder(path, escape=False, close_guard=False):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    inherited = []
    for entry in Path('/proc/self/fd').iterdir():
        try:
            inherited.append([int(entry.name), os.readlink(entry)])
        except FileNotFoundError:
            pass
    child = os.fork() if escape else 0
    if escape and child == 0:
        os.setsid()
        if close_guard:
            for fd, link in inherited:
                if fd > 2 and link.endswith('/accelerator.lock'):
                    os.close(fd)
        remaining = []
        for entry in Path('/proc/self/fd').iterdir():
            try:
                remaining.append(os.readlink(entry))
            except FileNotFoundError:
                pass
        Path(path + '.escaped').write_text(json.dumps({'pid': os.getpid(), 'fds': remaining}))
        while True:
            time.sleep(10)
    if escape:
        wait_for(path + '.escaped')
    Path(path).write_text(json.dumps({'pid': os.getpid(), 'escaped': child, 'fds': inherited,
        'limits': {name: resource.getrlimit(getattr(resource, 'RLIMIT_' + name))
                   for name in ('CORE', 'FSIZE', 'NOFILE')},
        'loader_overrides': sorted(key for key in os.environ if key.startswith(('LD_', 'PYTHON')))}))
    os.write(1, b'\x21\x00' * 320)
    while True:
        time.sleep(10)


def owner(namespace, marker):
    control = CaptureControl(time.monotonic() + 20, locked=lambda: False, namespace=namespace)
    value = CaptureOwner(control)
    stderr = tempfile.TemporaryFile()
    value.start([sys.executable, '-I', __file__, 'recorder', marker, 'escape', 'close'], stderr)
    return value, stderr


def observer(mode, root):
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    namespace, marker = str(root / 'leases'), str(root / 'recorder.json')
    unrelated = subprocess.Popen(['/bin/sleep', '30'])
    owned = []
    try:
        if mode in ('loss', 'loss-audio'):
            if mode == 'loss-audio':
                pcm = audio.MicCapture({'audio': {'capture_cmd': [sys.executable, '-I', __file__,
                    'recorder', marker, 'escape', 'close']}}, control=CaptureControl(
                        time.monotonic() + 20, locked=lambda: False, namespace=namespace))
                pcm.start()
                capture, stderr = pcm._owner, None
            else:
                capture, stderr = owner(namespace, marker)
            tree = wait_for(marker)
            escaped = wait_for(marker + '.escaped')
            assert not any(link.endswith('/accelerator.lock') for link in escaped['fds'])
            owned = [tree['pid'], tree['escaped']]
            os.kill(capture.process.pid, signal.SIGKILL)
            assert capture.wait(4) and not capture.stop()
            live = [Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[0] for pid in owned]
            assert all(state != 'Z' for state in live)
            assert capture.phase == 'unavailable'
            if mode == 'loss-audio':
                before_stop = time.monotonic()
                assert not pcm.stop() and time.monotonic() - before_stop < 1.5
                assert pcm.phase == 'unavailable' and pcm.error
            else:
                capture.process.stdout.close()
                stderr.close()
        else:
            producer = subprocess.Popen([sys.executable, '-I', __file__, 'producer', str(root)], stdout=subprocess.PIPE)
            supervisor = json.loads(producer.stdout.readline())['supervisor']
            tree = wait_for(marker)
            producer.kill(); producer.wait(timeout=2); producer.stdout.close()
            owned = [supervisor, tree['pid'], tree['escaped']]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    pid, _ = os.waitpid(supervisor, os.WNOHANG)
                except ChildProcessError:
                    pid = supervisor
                if pid:
                    break
                time.sleep(0.01)
            assert pid == supervisor
            assert not Path(f"/proc/{tree['pid']}").exists() and not Path(f"/proc/{tree['escaped']}").exists()
            live = []
        for pid in owned:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in owned:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        assert unavailable(namespace)
        assert unrelated.poll() is None
        print(json.dumps({'mode': mode, 'live_after_supervisor_loss': live, 'persistent_unavailable': True,
                          'loss_escaped_guard_absent': mode.startswith('loss'),
                          'unrelated_preserved': True, 'owned_absent': all(not Path(f'/proc/{pid}').exists() for pid in owned)}))
    finally:
        for pid in owned:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
        unrelated.terminate(); unrelated.wait(timeout=2)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'recorder':
        recorder(sys.argv[2], 'escape' in sys.argv[3:], 'close' in sys.argv[3:])
    elif mode == 'producer':
        root = Path(sys.argv[2])
        capture, stderr = owner(str(root / 'leases'), str(root / 'recorder.json'))
        print(json.dumps({'supervisor': capture.process.pid}), flush=True)
        time.sleep(30)
    elif mode in ('loss', 'loss-audio', 'parent-death'):
        observer(mode, Path(sys.argv[2]))
    else:
        raise SystemExit(2)
