"""Explicit managed capture, carried from Voice 869366f with cleanup proof.

The existing MicCapture default preserves published Whistle behavior. This
implementation is selected only by a typed CaptureControl; it has no model
runtime and delegates leases to their canonical component.
"""
from __future__ import annotations
import os
import queue
import select
import subprocess
import tempfile
import threading
import time
from typing import IO
from .audio import (MicCapture, AudioError, _positive_int, _EOF, _READ_CHUNK,
                    DEFAULT_RATE, DEFAULT_FRAME_MS, build_capture_cmd, _exit_message, _CAPTURE_HINT)
from .util import cfg_get
from .microphone import CaptureControl, CaptureOwner, MAX_SECONDS

class ManagedMicCapture(MicCapture):
    """One recorder subprocess plus one reader thread, in whole frames.

    The reader rebuffers the pipe — which delivers arbitrary chunk sizes — into
    exactly ``frame_bytes`` frames, so every consumer downstream (VAD, vosk)
    sees the frame length it was configured for and never a split sample.  The
    queue between the threads is bounded: a consumer that stalls loses the
    oldest audio rather than growing the process, because capture is realtime
    and stale frames are worth less than a live one.

    There is no pre-roll: nothing is recorded until ``start()``, which is the
    click-to-talk guarantee in DESIGN.md rather than an implementation detail.
    """

    _QUEUE_FRAMES = 512     # ~10 s at 20 ms
    _JOIN_TIMEOUT = 2.0
    _TERM_TIMEOUT = 1.0

    def __init__(self, cfg: dict | None = None, *, control: CaptureControl | None = None) -> None:
        self._cfg = cfg if cfg is not None else {}
        self._control = control
        self._owner: CaptureOwner | None = None
        self._rate = _positive_int(
            cfg_get(self._cfg, "audio.rate", DEFAULT_RATE), "audio.rate")
        frame_ms = _positive_int(
            cfg_get(self._cfg, "audio.frame_ms", DEFAULT_FRAME_MS),
            "audio.frame_ms")
        self._frame_bytes = self._rate * frame_ms // 1000 * 2
        if self._frame_bytes < 2:
            raise AudioError(
                f"audio.frame_ms {frame_ms} is shorter than one sample at "
                f"{self._rate} Hz. Use a frame length such as "
                f"{DEFAULT_FRAME_MS} ms.")
        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_FRAMES)
        self._stopping = threading.Event()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._started = False
        self._overruns = 0
        self._error: str | None = None

    @property
    def rate(self) -> int:
        """Capture sample rate in Hz."""
        return self._rate

    @property
    def frame_bytes(self) -> int:
        """Length of one frame in bytes: rate * frame_ms / 1000 * 2."""
        return self._frame_bytes

    @property
    def overruns(self) -> int:
        """Frames dropped because the consumer fell behind."""
        return self._overruns

    @property
    def error(self) -> str | None:
        """Why capture ended badly, or None. Consult this when read() ends."""
        if self._owner is not None and self._owner.error is not None:
            if self._owner.error.code not in ("cancelled", "deadline"):
                return str(self._owner.error)
        return self._error

    @property
    def phase(self) -> str:
        """Visible ownership state; unavailable is never an off indication."""
        return self._owner.phase if self._owner is not None else "inactive"

    @property
    def cleanup_complete(self) -> bool:
        return self._owner.cleanup_complete if self._owner is not None else True

    def start(self) -> None:
        """Launch the recorder and begin buffering frames."""
        if self._process is not None or not self.cleanup_complete:
            raise AudioError(
                "MicCapture.start() was called while capture was already "
                "running. Call stop() first; one MicCapture owns one recorder "
                "process.")
        cmd = build_capture_cmd(self._cfg)
        self._drain()
        self._stopping.clear()
        self._error = None
        self._overruns = 0
        # A temporary file rather than a pipe: the recorder's diagnostics are
        # only wanted if it dies, and a pipe nobody drains would wedge it.
        stderr = tempfile.TemporaryFile()
        try:
            control = self._control or CaptureControl(time.monotonic() + min(MAX_SECONDS,
                float(cfg_get(self._cfg, "stt.max_seconds", MAX_SECONDS))))
            self._owner = CaptureOwner(control)
            process = self._owner.start(cmd, stderr)
            self._process = process
            self._started = True
            self._reader = threading.Thread(
                target=self._read_stream, args=(process, stderr, cmd),
                name="kilix-voice-mic", daemon=True)
            self._reader.start()
        except BaseException as error:
            # Construction can fail after the recorder starts, and an
            # interruption can arrive while the reader itself is starting.
            # Keep ownership until teardown is proved in either case.
            try:
                self.stop()
            finally:
                stderr.close()
            if isinstance(error, Exception):
                raise AudioError(f"cannot start microphone capture: {error}") from error
            raise

    def read(self, timeout: float | None = None) -> bytes | None:
        """Return exactly one frame, or None on timeout or end of stream."""
        if not self._started:
            raise AudioError(
                "MicCapture.read() was called before start(). Call start() to "
                "open the microphone; a MicCapture holds no audio until then.")
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _EOF:
            # Put the sentinel back so every later read returns None at once
            # instead of blocking on a stream that has already ended.
            self._put(_EOF)
            return None
        return item

    def stop(self) -> bool:
        """Stop owned descendants; return only authenticated cleanup success."""
        self._stopping.set()
        process, reader = self._process, self._reader
        proven = self._owner.stop() if self._owner is not None else True
        if reader is not None and reader.ident is not None:
            reader.join(timeout=self._JOIN_TIMEOUT)
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        self._process = self._reader = None
        self._drain()
        # Guarantees a blocking read() after stop() returns rather than hangs,
        # even if the reader thread outlived its join timeout.
        self._put(_EOF)
        return proven

    def _read_stream(self, process: subprocess.Popen, stderr: IO[bytes],
                     cmd: list[str]) -> None:
        buffer = bytearray()
        want = self._frame_bytes
        stream = process.stdout
        try:
            poller = select.poll()
            poller.register(stream.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
            while not self._stopping.is_set():
                if not poller.poll(50):
                    if self._owner is not None and self._owner.wait(0):
                        break
                    continue
                chunk = os.read(stream.fileno(), _READ_CHUNK)
                if not chunk:
                    break
                buffer += chunk
                while len(buffer) >= want:
                    self._put(bytes(buffer[:want]))
                    del buffer[:want]
        except (OSError, ValueError):
            pass  # the pipe was closed under us by stop()
        # Whatever is left is a partial frame: dropped, not padded, because a
        # short frame would shift every following sample in the recogniser.
        try:
            code = process.wait(timeout=self._TERM_TIMEOUT)
        except subprocess.TimeoutExpired:
            code = None
        if self._owner is not None:
            self._owner.wait(3.5)
        if (not self._stopping.is_set() and code not in (0, None)
                and (self._owner is None or self._owner.error is None)):
            self._error = _exit_message("capture", cmd, code, stderr,
                                        _CAPTURE_HINT)
        stderr.close()
        self._put(_EOF)

    def _put(self, item: object) -> None:
        """Queue one frame, dropping the oldest when the consumer is behind."""
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._overruns += 1
                except queue.Empty:
                    pass

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
