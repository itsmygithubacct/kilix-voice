"""AUD-02 / V19: caller-facing audio input by descriptor, op ingest-audio.

Every exchange goes over a real AF_UNIX SOCK_SEQPACKET listener served by the
real Daemon._accept_one, with the audio attached by SCM_RIGHTS. After each
one, no descriptor the daemon received or made is left open on its side, and
no worker thread was started.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_ingest", importlib.machinery.SourceFileLoader(
        "kilix_voiced_ingest", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import audiofd, jobs, protocol, util  # noqa: E402

from tests.livedaemon import LiveDaemonTestCase  # noqa: E402

_INT = struct.calcsize("i")
PCM = bytes((i * 13) & 0xFF for i in range(3200))          # 100 ms at 16 kHz


def wav_bytes(pcm: bytes, rate: int = 16000, channels: int = 1, bits: int = 16,
              tag: int = 1) -> bytes:
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    body = (b"fmt " + struct.pack("<I", 16) + fmt + b"data"
            + struct.pack("<I", len(pcm)) + pcm)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def _received(ancillary) -> list[int]:
    return [fd for level, kind, blob in ancillary
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS
            for fd in struct.unpack(f"{len(blob) // _INT}i", blob)]


def _open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


class _IngestFixture(unittest.TestCase):

    def setUp(self) -> None:
        self.session = os.path.realpath(tempfile.mkdtemp(prefix="kv-ingest-", dir="/tmp"))
        self.addCleanup(shutil.rmtree, self.session, True)
        self.path = os.path.join(self.session, "control.sock")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(listener.close)
        listener.bind(self.path)
        listener.listen(4)
        listener.setblocking(False)
        self.warnings = []
        d = object.__new__(voiced.Daemon)
        d._socket = listener
        d._session_dir = self.session
        d._stopping = threading.Event()
        d._lock = threading.RLock()
        d._turns = 0
        d._jobs = jobs.JobLedger()
        d._touch = lambda: None
        d._warn = self.warnings.append
        d._debug = lambda *a, **k: None
        d._start = mock.Mock(side_effect=AssertionError("ingest must not start a worker"))
        self.daemon = d

    def pipe_with(self, data: bytes, close_writer: bool = True) -> int:
        read_end, write_end = os.pipe()
        os.write(write_end, data)
        if close_writer:
            os.close(write_end)
        else:
            self.addCleanup(os.close, write_end)
        return read_end

    def memfd_with(self, data: bytes) -> int:
        fd = os.memfd_create("kv-ingest-test")
        os.write(fd, data)
        os.lseek(fd, 0, os.SEEK_SET)
        return fd

    def file_with(self, data: bytes) -> int:
        path = os.path.join(self.session, f"input-{len(os.listdir(self.session))}.bin")
        with open(path, "wb") as handle:
            handle.write(data)
        return os.open(path, os.O_RDONLY)

    def exchange(self, request: dict, fds=()) -> tuple[dict, list[int]]:
        threads = threading.active_count()
        before = _open_descriptors()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(5)
        client.connect(self.path)
        frame = protocol.encode(dict(request, op="ingest-audio"))
        if fds:
            client.sendmsg([frame], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                      struct.pack(f"{len(fds)}i", *fds))])
        else:
            client.send(frame)
        for fd in fds:
            os.close(fd)                      # the caller's own copies
        voiced.Daemon._accept_one(self.daemon)
        data, ancillary, _flags, _ = client.recvmsg(1 << 20, socket.CMSG_SPACE(4 * _INT))
        client.close()
        received = _received(ancillary)
        for fd in received:
            self.addCleanup(os.close, fd)
        self.assertEqual(_open_descriptors(), before - len(fds) + len(received),
                         "a descriptor the daemon received or made was left open")
        self.assertEqual(threading.active_count(), threads)
        self.daemon._start.assert_not_called()
        return protocol.decode(data), received

    def assert_refused(self, request: dict, fds, code: str) -> dict:
        reply, received = self.exchange(request, fds)
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], code, reply)
        self.assertEqual(received, [])
        return reply


class IngestAccepted(_IngestFixture):

    def test_a_prefilled_pipe_wav_comes_back_as_a_sealed_canonical_wav(self) -> None:
        reply, fds = self.exchange({"id": "a", "sample_format": "s16le", "channels": 1,
                                    "sample_rate": 16000, "duration_limit_ms": 1000,
                                    "byte_limit": 65536},
                                   [self.pipe_with(wav_bytes(PCM))])
        self.assertIs(reply["ok"], True, reply)
        self.assertEqual(len(fds), 1)
        got = os.pread(fds[0], 1 << 20, 0)
        self.assertEqual(got, util.write_wav(PCM, 16000))
        self.assertEqual((reply["pcm_bytes"], reply["byte_length"], reply["duration_ms"],
                          reply["sample_rate"]), (3200, 3244, 100, 16000))
        self.assertEqual(reply["sha256"], hashlib.sha256(got).hexdigest())
        self.assertEqual((reply["audio_fd"], reply["media_type"], reply["sample_format"],
                          reply["channels"], reply["id"]), (0, "audio/wav", "s16le", 1, "a"))
        with self.assertRaises(OSError):
            os.write(fds[0], b"x")
        seals = fcntl.fcntl(fds[0], audiofd.F_GET_SEALS)
        for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL"):
            # Each flag by name: the aggregate constant is what a broken
            # sealing step would have been built from.
            self.assertTrue(seals & getattr(audiofd, name), f"{name} is missing")
        writable = os.open(f"/proc/self/fd/{fds[0]}", os.O_WRONLY)
        try:
            with self.assertRaises(PermissionError):
                os.write(writable, b"x")
        finally:
            os.close(writable)
        record = self.daemon._jobs.get(reply["job"])
        self.assertTrue(reply["job"].startswith("ingest-"))
        self.assertEqual((record["kind"], record["outcome"]), ("ingest", "completed"))

    def test_a_deadline_that_passes_while_normalising_returns_nothing(self) -> None:
        # The delivery boundary: audio read in time but ready only after the
        # caller's budget ran out is not handed back as a success.
        instants = iter([100.0, 100.0] + [200.0] * 20)
        # A memfd, not a pipe: a pipe read consults the deadline itself, on
        # the real clock, and would refuse before normalising -- the test
        # would then pass without ever reaching the boundary it is about.
        read_end = self.memfd_with(wav_bytes(PCM))
        self.addCleanup(os.close, read_end)
        before = _open_descriptors()
        with mock.patch.object(voiced.time, "monotonic", lambda: next(instants)):
            reply = voiced.Daemon._op_ingest_audio(
                self.daemon, {"op": "ingest-audio", "id": "", "deadline_ms": 5000},
                (read_end,), False)
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)
        self.assertEqual(getattr(reply, "fds", ()), ())
        self.assertEqual(_open_descriptors(), before, "a sealed copy was left open")
        self.assertEqual(self.daemon._jobs.recent(), [])

    def test_raw_s16le_on_a_memfd_with_a_declared_rate(self) -> None:
        for request in ({"sample_rate": 22050, "container": "raw"}, {"sample_rate": 22050}):
            with self.subTest(request=request):
                reply, fds = self.exchange(request, [self.memfd_with(PCM)])
                self.assertIs(reply["ok"], True, reply)
                self.assertEqual(util.parse_wav_bytes(os.pread(fds[0], 1 << 20, 0)),
                                 (PCM, 22050))
                self.assertEqual(reply["duration_ms"], 73)      # ceil(1600 / 22050 s)

    def test_a_regular_file_opened_read_only(self) -> None:
        reply, fds = self.exchange({}, [self.file_with(wav_bytes(PCM, rate=8000))])
        self.assertIs(reply["ok"], True, reply)
        self.assertEqual(util.parse_wav_bytes(os.pread(fds[0], 1 << 20, 0)), (PCM, 8000))


class IngestRefused(_IngestFixture):

    def test_no_descriptor_is_malformed(self) -> None:
        self.assert_refused({}, (), protocol.ERR_MALFORMED)

    def test_two_descriptors_are_malformed_and_both_closed(self) -> None:
        pipes = [os.pipe(), os.pipe()]
        for _read, write in pipes:
            self.addCleanup(os.close, write)
        self.assert_refused({}, [read for read, _ in pipes], protocol.ERR_MALFORMED)
        for _read, write in pipes:
            with self.assertRaises(BrokenPipeError):
                os.write(write, b"x")

    def counting(self, name: str):
        real = getattr(os, name)
        calls = []

        def wrapper(*args):
            result = real(*args)
            calls.append(len(result))
            return result

        return mock.patch.object(audiofd.os, name, wrapper), calls

    def test_a_character_device_is_unsupported_and_never_read(self) -> None:
        read_patch, reads = self.counting("read")
        pread_patch, preads = self.counting("pread")
        with read_patch, pread_patch:
            self.assert_refused({}, [os.open("/dev/zero", os.O_RDONLY)],
                                protocol.ERR_UNSUPPORTED)
        self.assertEqual((reads, preads), ([], []))

    def test_a_socket_is_unsupported(self) -> None:
        a, b = socket.socketpair()
        self.addCleanup(b.close)
        self.assert_refused({}, [a.detach()], protocol.ERR_UNSUPPORTED)

    def test_a_pipe_over_the_byte_limit_is_too_large_and_not_drained(self) -> None:
        read_patch, reads = self.counting("read")
        with read_patch:
            self.assert_refused({"byte_limit": 100, "container": "raw", "sample_rate": 8000},
                                [self.pipe_with(b"\x00" * 1000)], protocol.ERR_TOO_LARGE)
        self.assertLessEqual(sum(reads), 101)

    def test_a_file_over_the_byte_limit_is_too_large_without_a_read(self) -> None:
        pread_patch, preads = self.counting("pread")
        with pread_patch:
            self.assert_refused({"byte_limit": 100}, [self.file_with(b"\x00" * 101)],
                                protocol.ERR_TOO_LARGE)
        self.assertEqual(preads, [])

    def test_a_wav_that_is_not_16_bit_mono_is_unsupported(self) -> None:
        for shape in ({"bits": 24}, {"channels": 2}, {"tag": 3}):
            with self.subTest(**shape):
                pcm = PCM * 3 if shape.get("bits") == 24 else PCM
                self.assert_refused({}, [self.pipe_with(wav_bytes(pcm, **shape))],
                                    protocol.ERR_UNSUPPORTED)

    def test_a_declared_rate_the_wav_does_not_have_is_malformed(self) -> None:
        self.assert_refused({"sample_rate": 22050}, [self.pipe_with(wav_bytes(PCM))],
                            protocol.ERR_MALFORMED)

    def test_raw_pcm_without_a_rate_or_whole_samples_is_malformed(self) -> None:
        self.assert_refused({"container": "raw"}, [self.pipe_with(PCM)], protocol.ERR_MALFORMED)
        self.assert_refused({"sample_rate": 16000, "container": "raw"},
                            [self.pipe_with(PCM + b"\x01")], protocol.ERR_MALFORMED)

    def test_a_truncated_wav_is_malformed_not_trimmed(self) -> None:
        self.assert_refused({}, [self.pipe_with(wav_bytes(PCM)[:-100])], protocol.ERR_MALFORMED)

    def test_audio_longer_than_its_duration_limit_is_too_large(self) -> None:
        self.assert_refused({"duration_limit_ms": 50}, [self.pipe_with(wav_bytes(PCM))],
                            protocol.ERR_TOO_LARGE)

    def test_a_pipe_whose_writer_stays_open_is_refused_within_its_window(self) -> None:
        started = time.monotonic()
        self.assert_refused({}, [self.pipe_with(b"RIFF", close_writer=False)],
                            protocol.ERR_MALFORMED)
        self.assertLess(time.monotonic() - started, audiofd.INGEST_READ_TIMEOUT_S + 0.3)
        started = time.monotonic()
        self.assert_refused({"deadline_ms": 100}, [self.pipe_with(b"RIFF", close_writer=False)],
                            protocol.ERR_DEADLINE)
        self.assertLess(time.monotonic() - started, 0.3)

    def test_a_spent_deadline_never_reads_the_descriptor(self) -> None:
        instants = iter([100.0] + [200.0] * 20)
        reader = mock.Mock(side_effect=AssertionError("the descriptor was read"))
        read_end = self.pipe_with(wav_bytes(PCM))
        self.addCleanup(os.close, read_end)
        with mock.patch.object(voiced.time, "monotonic", lambda: next(instants)), \
             mock.patch.object(voiced.audiofd, "read_descriptor", reader):
            reply = voiced.Daemon._op_ingest_audio(
                self.daemon, {"op": "ingest-audio", "id": "", "deadline_ms": 1},
                (read_end,), False)
        self.assertEqual(reply["code"], protocol.ERR_DEADLINE, reply)
        reader.assert_not_called()


def attempts(name: str):
    """Record every call to os.<name> as it is MADE, so one that raises counts."""
    real = getattr(os, name)
    calls = []

    def wrapper(*args, **kwargs):
        calls.append(args[0] if args else None)
        return real(*args, **kwargs)

    return mock.patch.object(audiofd.os, name, wrapper), calls


def reopens(calls) -> list:
    return [c for c in calls if isinstance(c, str) and c.startswith("/proc/self/fd/")]


class IngestDescriptorAccess(_IngestFixture):
    """A descriptor that grants no read is refused before anything reads it.

    Reopening through /proc/self/fd would read an O_PATH file or a pipe's
    write end that the caller never opened for reading, so the checks come
    before any read or reopen, and these tests watch for both.
    """

    def assert_refused_unread(self, fd: int) -> None:
        pread_patch, preads = attempts("pread")
        read_patch, reads = attempts("read")
        open_patch, opens = attempts("open")
        with pread_patch, read_patch, open_patch:
            self.assert_refused({}, [fd], protocol.ERR_MALFORMED)
        self.assertEqual((preads, reads, reopens(opens)), ([], [], []))

    def test_an_o_path_descriptor_is_malformed_and_never_read(self) -> None:
        regular = self.file_with(wav_bytes(PCM))
        o_path = os.open(f"/proc/self/fd/{regular}", os.O_PATH)
        os.close(regular)
        with self.subTest(kind="regular file"):
            self.assert_refused_unread(o_path)
        read_end = self.pipe_with(wav_bytes(PCM))
        pipe_path = os.open(f"/proc/self/fd/{read_end}", os.O_PATH)
        self.addCleanup(os.close, read_end)
        with self.subTest(kind="pipe"):
            self.assert_refused_unread(pipe_path)

    def test_a_write_only_descriptor_is_malformed_and_never_read(self) -> None:
        path = os.path.join(self.session, "write-only.wav")
        with open(path, "wb") as handle:
            handle.write(wav_bytes(PCM))
        with self.subTest(kind="regular file"):
            self.assert_refused_unread(os.open(path, os.O_WRONLY))
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, read_end)
        os.write(write_end, wav_bytes(PCM))
        with self.subTest(kind="pipe write end"):
            self.assert_refused_unread(write_end)


class _AlwaysReady:
    """A poll that reports the pipe readable every time: the sender won the race."""

    def register(self, *args) -> None:
        pass

    def poll(self, timeout=None):
        time.sleep(0.005)
        return [(0, 1)]


class IngestPipeRace(unittest.TestCase):
    """AUD-02: the sender shares the pipe, and may take the bytes first.

    Readiness can be stale by the time the read runs: the sender kept its own
    read end and drained what the check saw. A blocking read of the shared
    description then waited for as long as the sender held its writer open,
    in the daemon's one accept loop. Here the race is forced every time.
    """

    def read_in_thread(self, fd: int) -> dict:
        box: dict = {}

        def run() -> None:
            try:
                box["data"] = audiofd.read_descriptor(fd, byte_limit=65536)
            except BaseException as error:
                box["error"] = error

        with mock.patch.object(audiofd.select, "poll", _AlwaysReady):
            reader = threading.Thread(target=run, daemon=True)
            reader.start()
            reader.join(audiofd.INGEST_READ_TIMEOUT_S + 2.0)
        self.assertFalse(reader.is_alive(), "the read blocked on the sender's pipe")
        return box

    def setUp(self) -> None:
        self.writers: set[int] = set()
        self.addCleanup(self.close_writers)

    def close_writers(self) -> None:
        # Closing a writer ends any read still blocked on it, even a broken one's.
        for fd in sorted(self.writers):
            os.close(fd)
        self.writers.clear()

    def pipe(self) -> tuple[int, int]:
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, read_end)
        self.writers.add(write_end)
        return read_end, write_end

    def test_a_sender_that_holds_and_drains_the_pipe_gets_a_bounded_refusal(self) -> None:
        read_end, _write_end = self.pipe()
        flags = fcntl.fcntl(read_end, fcntl.F_GETFL)
        started = time.monotonic()
        box = self.read_in_thread(read_end)
        self.assertIsInstance(box.get("error"), audiofd.DescriptorError, box)
        self.assertEqual(box["error"].code, protocol.ERR_MALFORMED)
        self.assertLess(time.monotonic() - started, audiofd.INGEST_READ_TIMEOUT_S + 0.5)
        self.assertEqual(fcntl.fcntl(read_end, fcntl.F_GETFL), flags,
                         "the sender's own end of the pipe was changed")

    def test_a_sender_that_set_o_nonblock_gets_a_code_not_a_fault(self) -> None:
        read_end, write_end = self.pipe()
        flags = fcntl.fcntl(read_end, fcntl.F_GETFL) | os.O_NONBLOCK
        fcntl.fcntl(read_end, fcntl.F_SETFL, flags)
        box = self.read_in_thread(read_end)
        self.assertIsInstance(box.get("error"), audiofd.DescriptorError, box)
        self.assertEqual(box["error"].code, protocol.ERR_MALFORMED)
        os.write(write_end, b"all of it")                      # control
        self.writers.discard(write_end)
        os.close(write_end)
        self.assertEqual(self.read_in_thread(read_end), {"data": b"all of it"})
        self.assertEqual(fcntl.fcntl(read_end, fcntl.F_GETFL), flags)

    def test_a_read_the_system_refuses_carries_a_code(self) -> None:
        for number, code in ((errno.EIO, protocol.ERR_UNAVAILABLE),
                             (errno.EBADF, protocol.ERR_MALFORMED)):
            with self.subTest(errno=errno.errorcode[number]):
                fd = os.memfd_create("kv-ingest-eio")
                self.addCleanup(os.close, fd)
                os.write(fd, wav_bytes(PCM))
                failing = mock.Mock(side_effect=OSError(number, os.strerror(number)))
                with mock.patch.object(audiofd.os, "pread", failing):
                    with self.assertRaises(audiofd.DescriptorError) as caught:
                        audiofd.read_descriptor(fd, byte_limit=65536)
                self.assertEqual(caught.exception.code, code)


class IngestStalledDescriptor(_IngestFixture):
    """A descriptor whose inspection never answers cannot hold the accept loop.

    A file on a filesystem the caller serves can hold fstat or pread for as
    long as it likes. Here fstat is held until the test releases it.
    """

    def stalled_exchange(self, request: dict) -> tuple[dict, float]:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(5)
        with client:
            client.connect(self.path)
            fd = self.memfd_with(wav_bytes(PCM))
            client.sendmsg([protocol.encode(dict(request, op="ingest-audio"))],
                           [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))])
            os.close(fd)
            started = time.monotonic()
            voiced.Daemon._accept_one(self.daemon)
            elapsed = time.monotonic() - started
            data, ancillary, _flags, _ = client.recvmsg(1 << 16, socket.CMSG_SPACE(4 * _INT))
        for received in _received(ancillary):
            os.close(received)
        return protocol.decode(data), elapsed

    def test_a_stalled_descriptor_is_refused_in_bound_and_stalls_are_capped(self) -> None:
        release, entered = threading.Event(), []
        real_fstat = os.fstat

        def held_fstat(fd, *args, **kwargs):
            if threading.current_thread().name == "kilix-voice-ingest-read":
                entered.append(fd)
                release.wait(20)
            return real_fstat(fd, *args, **kwargs)

        self.addCleanup(release.set)
        bound = audiofd.INGEST_READ_TIMEOUT_S + audiofd._READ_GRACE_S + 0.3
        with mock.patch.object(audiofd.os, "fstat", held_fstat):
            reply, elapsed = self.stalled_exchange({"id": "one"})
            self.assertEqual(reply.get("code"), protocol.ERR_UNAVAILABLE, reply)
            self.assertLess(elapsed, bound)
            reply, elapsed = self.stalled_exchange({"id": "two", "deadline_ms": 200})
            self.assertEqual(reply.get("code"), protocol.ERR_DEADLINE, reply)
            self.assertLess(elapsed, 0.2 + audiofd._READ_GRACE_S + 0.3)
            self.assertEqual(len(entered), audiofd.MAX_STALLED_READS)
            reply, elapsed = self.stalled_exchange({"id": "three"})
            self.assertEqual(reply.get("code"), protocol.ERR_BUSY, reply)
            self.assertLess(elapsed, 0.3)
            self.assertEqual(len(entered), audiofd.MAX_STALLED_READS,
                             "a descriptor was inspected past the stall cap")
            release.set()
            # Once the stuck reads end, their slots come back.
            self.assertTrue(_eventually(lambda: self.stalled_exchange({"id": "four"})[0]
                                        .get("ok") is True, 5))


def _eventually(predicate, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class IngestLiveControlPlane(LiveDaemonTestCase):
    """A real kilix-voiced: a hostile ingest pipe cannot hold stop-speech or status."""

    SETTINGS = ("KILIX_VOICE_TTS_ENGINE=off\n"
                "KILIX_VOICE_STT_ENGINE=vosk\n"
                "KILIX_VOICE_STT_MODEL=small-en-us\n")
    HOLD_S = 4.0

    def ingest(self, fd: int, box: dict) -> None:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(15)
        with client:
            client.connect(self.control)
            started = time.monotonic()
            client.sendmsg([b'{"op":"ingest-audio","id":"hostile"}\n'],
                           [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))])
            data, ancillary, _flags, _ = client.recvmsg(1 << 16, socket.CMSG_SPACE(4 * _INT))
            box["elapsed"] = time.monotonic() - started
        for received in _received(ancillary):
            os.close(received)
        box["reply"] = protocol.decode(data)

    def test_a_pipe_its_sender_holds_and_drains_never_holds_the_control_plane(self) -> None:
        read_end, write_end = os.pipe()
        stop, writer_closed = threading.Event(), threading.Event()

        def writer() -> None:
            # ONE byte, once the daemon is waiting on the pipe, and then the
            # writer is held open with nothing more written. When the drainer
            # wins that byte, a read of the shared description has nothing to
            # return until the writer closes. The writer closes itself after
            # HOLD_S: a stalled daemon stalls this test's own probes too.
            time.sleep(0.2)
            os.write(write_end, b"R")
            stop.wait(self.HOLD_S)
            os.close(write_end)
            writer_closed.set()

        def drainer() -> None:
            while not stop.is_set():
                try:
                    if not os.read(read_end, 1):
                        return
                except OSError:
                    return

        helpers = [threading.Thread(target=writer, daemon=True),
                   threading.Thread(target=drainer, daemon=True)]
        for helper in helpers:
            helper.start()
        box: dict = {}
        ingest = threading.Thread(target=self.ingest, args=(read_end, box), daemon=True)
        ingest.start()
        latencies = []
        end = time.monotonic() + self.HOLD_S
        try:
            while time.monotonic() < end:
                for op in ("stop-speech", "status"):
                    started = time.monotonic()
                    reply = self.request({"op": op})
                    latencies.append((time.monotonic() - started, op))
                    self.assertTrue(reply["ok"], reply)
                time.sleep(0.05)
            ingest.join(5)
        finally:
            stop.set()                  # the writer closes; a blocked read sees end-of-file
            self.assertTrue(writer_closed.wait(self.HOLD_S + 5))
            for helper in helpers:
                helper.join(2)
            os.close(read_end)
        bound = audiofd.INGEST_READ_TIMEOUT_S + audiofd._READ_GRACE_S + 0.5
        self.assertFalse(ingest.is_alive(), "the ingest request was never answered")
        self.assertEqual(box["reply"].get("code"), protocol.ERR_MALFORMED, box)
        self.assertLess(box["elapsed"], bound, box)
        worst, op = max(latencies)
        self.assertLess(worst, bound, f"{op} waited {worst:.3f} s: the control plane was held")
        self.assert_still_serving()


class IngestRequestValidation(unittest.TestCase):
    """The protocol side, including the V19 vector's frozen constraints."""

    def setUp(self) -> None:
        self.session = os.path.realpath(tempfile.mkdtemp(prefix="kv-ingest-v-"))
        self.addCleanup(shutil.rmtree, self.session, True)

    def validate(self, message: dict) -> dict:
        return protocol.validate_request(dict(message, op="ingest-audio"), self.session)

    def test_the_op_validates_with_only_op_and_deadline(self) -> None:
        self.assertEqual(self.validate({"deadline_ms": 5000}),
                         {"op": "ingest-audio", "id": "", "deadline_ms": 5000})

    def test_v19s_candidate_fields_never_raise(self) -> None:
        import base64
        preview = base64.b64encode(wav_bytes(bytes(3200))).decode("ascii")
        candidates = {"audio_fd": 3, "fd": 3, "audio": preview, "audio_path": "/x.wav",
                      "input": "/x.wav", "path": "/x.wav", "file": "/x.wav", "wav": preview,
                      "pcm": preview, "stream": "/x.wav", "sample_format": "s16le",
                      "sample_rate": 16000, "channels": 1, "duration_limit_ms": 1000,
                      "byte_limit": 65536}
        for op in protocol.OPS:
            with self.subTest(op=op):
                message = {"op": op, **candidates}
                if op == "speak":
                    message["text"] = "hi"
                if op == "dictate":
                    message["sock"] = os.path.join(self.session, "dictate-1.sock")
                protocol.validate_request(message, self.session)     # must not raise
        kept = self.validate(candidates)
        self.assertEqual(set(kept) - {"op", "id"},
                         {"sample_format", "sample_rate", "channels",
                          "duration_limit_ms", "byte_limit"})

    def test_each_declaration_is_refused_with_its_code(self) -> None:
        table = (
            ({"sample_format": "f32le"}, protocol.ERR_UNSUPPORTED),
            ({"sample_format": 5}, protocol.ERR_MALFORMED),
            ({"channels": 2}, protocol.ERR_UNSUPPORTED),
            ({"channels": "1"}, protocol.ERR_MALFORMED),
            ({"sample_rate": 96000}, protocol.ERR_UNSUPPORTED),
            ({"sample_rate": 7999}, protocol.ERR_UNSUPPORTED),
            ({"sample_rate": True}, protocol.ERR_MALFORMED),
            ({"container": "flac"}, protocol.ERR_UNSUPPORTED),
            ({"container": None}, protocol.ERR_MALFORMED),
            ({"duration_limit_ms": 0}, protocol.ERR_MALFORMED),
            ({"duration_limit_ms": protocol.MAX_INGEST_DURATION_MS + 1}, protocol.ERR_MALFORMED),
            ({"byte_limit": 0}, protocol.ERR_MALFORMED),
            ({"byte_limit": protocol.MAX_AUDIO_BYTES + 1}, protocol.ERR_MALFORMED),
        )
        for field, code in table:
            with self.subTest(field=field):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self.validate(field)
                self.assertEqual(caught.exception.code or protocol.ERR_MALFORMED, code)

    def test_other_ops_drop_the_declarations(self) -> None:
        request = protocol.validate_request(
            {"op": "status", "sample_rate": 16000, "byte_limit": 5, "container": "raw"},
            self.session)
        self.assertEqual(request, {"op": "status", "id": ""})


if __name__ == "__main__":
    unittest.main()
