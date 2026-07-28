"""Microphone capture and PCM playback through external command-line tools.

kilix-voice links no audio library: recording and playback are ordinary
subprocesses (``parec``/``arecord``, ``pacat``/``aplay``) exchanging raw s16le
mono PCM over a pipe.  That keeps the package stdlib-only, keeps the audio
device owned by one short-lived process that dies with its clip, and makes the
whole module testable with no audio server present — ``build_capture_cmd`` and
``build_play_cmd`` are pure functions of the config that launch nothing, and a
test injects a fake recorder or sink through the command overrides.

Config keys, all optional, as dotted paths into the daemon's config dict:

===================  ==========================================  ===========
``audio.rate``       capture sample rate in Hz                    ``16000``
``audio.frame_ms``   capture frame length in milliseconds         ``20``
``audio.device_in``  PulseAudio source token, or ``"default"``    ``default``
``audio.device_out`` PulseAudio sink token, or ``"default"``      ``default``
``audio.capture_cmd`` argv list replacing the detected recorder   unset
``audio.play_cmd``   argv list replacing the detected sink        unset
===================  ==========================================  ===========

Both overrides substitute ``{rate}`` and ``{device}`` in every argument and are
executed directly — never through a shell.  The daemon fills the device keys
from ``KILIX_VOICE_DEVICE_IN``/``_OUT``; this module reads no settings file, so
that command building stays a pure function of what it was handed.

Nothing here opens a device before ``start()``/``play()``: constructing a
``MicCapture`` or a ``Player`` is free, which is what lets a curses TUI build
one at import time and only then decide whether to use it.
"""

from __future__ import annotations

import queue
import re
import subprocess
import tempfile
import threading
from typing import IO

from .util import cfg_get, which

DEFAULT_RATE = 16000
DEFAULT_FRAME_MS = 20
DEFAULT_DEVICE = "default"

# Read granularity for the capture pipe. read1() returns as soon as anything is
# available, so a large ceiling costs latency nothing and syscalls something.
_READ_CHUNK = 65536

_EOF = object()  # queue sentinel: the capture stream ended

# Device tokens become argv for parec/pacat, so they are held to the same
# alphabet settings.py accepts rather than passed through as arbitrary text.
_DEVICE_TOKEN = re.compile(r"^[A-Za-z0-9_.:+-]{1,128}$")

_CAPTURE_HINT = (
    "Check that a microphone is connected and that PulseAudio or PipeWire is "
    "running (`pactl info`), or set KILIX_VOICE_DEVICE_IN to a source listed "
    "by `pactl list sources short`.")
_PLAYBACK_HINT = (
    "Check that PulseAudio or PipeWire is running (`pactl info`), or set "
    "KILIX_VOICE_DEVICE_OUT to a sink listed by `pactl list sinks short`.")


class AudioError(RuntimeError):
    """An audio subsystem failure; the message says what to install or fix."""


def _positive_int(raw: object, what: str) -> int:
    """Return ``raw`` as a positive int, or raise AudioError naming ``what``."""
    try:
        number = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise AudioError(
            f"{what} must be a whole number, got {raw!r}. Set it to a number "
            f"of Hz such as {DEFAULT_RATE}.") from error
    if number <= 0:
        raise AudioError(
            f"{what} must be greater than zero, got {number}. Set it to a "
            f"number of Hz such as {DEFAULT_RATE}.")
    return number


def _device(cfg, path: str) -> str:
    """Return a validated device token from ``cfg``, or ``"default"``."""
    token = str(cfg_get(cfg, path, DEFAULT_DEVICE) or DEFAULT_DEVICE).strip()
    if token == DEFAULT_DEVICE:
        return DEFAULT_DEVICE
    if not _DEVICE_TOKEN.match(token):
        raise AudioError(
            f"{path} is not a usable device name: {token!r}. Use 'default', or "
            "a token from `pactl list sources short` / `pactl list sinks "
            "short` (letters, digits and ._:+- only).")
    return token


def _argv(template: object, key: str, rate: int, device: str) -> list[str]:
    """Return an override command with ``{rate}``/``{device}`` substituted."""
    if isinstance(template, (str, bytes)) or not isinstance(
            template, (list, tuple)):
        raise AudioError(
            f"{key} must be a list of arguments, got "
            f"{type(template).__name__}. Write it as a list, for example "
            '["cat", "fixture.raw"]; the command is executed directly, never '
            "through a shell.")
    argv = [str(part).replace("{rate}", str(rate)).replace("{device}", device)
            for part in template]
    if not argv or not argv[0]:
        raise AudioError(
            f"{key} names no program. Put the executable first, for example "
            '["parec", "--format=s16le"], or remove the key to auto-detect '
            "the tools installed on this machine.")
    return argv


def build_capture_cmd(cfg) -> list[str]:
    """Return the command that writes raw s16le mono capture to stdout.

    Pure: it inspects the config and PATH and launches nothing, so a test can
    assert on the exact argv with no audio server anywhere on the machine.
    """
    rate = _positive_int(cfg_get(cfg, "audio.rate", DEFAULT_RATE), "audio.rate")
    device = _device(cfg, "audio.device_in")
    override = cfg_get(cfg, "audio.capture_cmd")
    if override is not None:
        return _argv(override, "audio.capture_cmd", rate, device)
    if which("parec"):
        cmd = ["parec", "--format=s16le", f"--rate={rate}", "--channels=1",
               "--latency-msec=30"]
        if device != DEFAULT_DEVICE:
            cmd += ["-d", device]
        return cmd
    if which("arecord"):
        return ["arecord", "-q", "-f", "S16_LE", "-r", str(rate), "-c", "1",
                "-t", "raw"]
    raise AudioError(
        "no microphone capture tool found. Install pulseaudio-utils (parec) "
        "or alsa-utils (arecord) — on Debian: sudo apt install "
        "pulseaudio-utils — or set audio.capture_cmd to a command that writes "
        "raw s16le mono audio to stdout.")


def build_play_cmd(cfg, rate: int) -> list[str]:
    """Return the command that plays raw s16le mono at ``rate`` from stdin.

    The rate is per clip rather than per config: TTS engines synthesise at
    their own native rate and the sink is told which one this clip is.
    """
    rate = _positive_int(rate, "the clip sample rate")
    device = _device(cfg, "audio.device_out")
    override = cfg_get(cfg, "audio.play_cmd")
    if override is not None:
        return _argv(override, "audio.play_cmd", rate, device)
    if which("pacat"):
        cmd = ["pacat", "--playback", "--format=s16le", f"--rate={rate}",
               "--channels=1"]
        if device != DEFAULT_DEVICE:
            cmd += ["-d", device]
        return cmd
    if which("aplay"):
        return ["aplay", "-q", "-f", "S16_LE", "-r", str(rate), "-c", "1",
                "-t", "raw"]
    raise AudioError(
        "no audio playback tool found. Install pulseaudio-utils (pacat) or "
        "alsa-utils (aplay) — on Debian: sudo apt install pulseaudio-utils — "
        "or set audio.play_cmd to a command that reads raw s16le mono audio "
        "from stdin.")


def _stderr_tail(stream: IO[bytes], limit: int = 300) -> str:
    """Return the tail of a subprocess's captured stderr as one line."""
    try:
        stream.seek(0)
        text = stream.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    return " ".join(text.split())[-limit:]


def _exit_message(what: str, cmd: list[str], code: int | None,
                  stderr: IO[bytes], hint: str) -> str:
    """Return an actionable message for a helper process that failed."""
    detail = _stderr_tail(stderr)
    return (f"the {what} command {cmd[0]!r} exited with status {code}"
            f"{': ' + detail if detail else ''}. {hint}")


class MicCapture:
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

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg if cfg is not None else {}
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
        return self._error

    def start(self) -> None:
        """Launch the recorder and begin buffering frames."""
        if self._process is not None:
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
            process = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=stderr)
        except OSError as error:
            stderr.close()
            raise AudioError(
                f"cannot run the capture command {cmd[0]!r}: {error}. Install "
                "that tool, or set audio.capture_cmd to a command that writes "
                "raw s16le mono audio to stdout.") from error
        self._process = process
        self._started = True
        self._reader = threading.Thread(
            target=self._read_stream, args=(process, stderr, cmd),
            name="kilix-voice-mic", daemon=True)
        self._reader.start()

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

    def stop(self) -> None:
        """Stop the recorder and join the reader. Idempotent."""
        self._stopping.set()
        process, self._process = self._process, None
        reader, self._reader = self._reader, None
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=self._TERM_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except OSError:
                pass
        if reader is not None:
            reader.join(timeout=self._JOIN_TIMEOUT)
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        # Guarantees a blocking read() after stop() returns rather than hangs,
        # even if the reader thread outlived its join timeout.
        self._put(_EOF)

    def _read_stream(self, process: subprocess.Popen, stderr: IO[bytes],
                     cmd: list[str]) -> None:
        buffer = bytearray()
        want = self._frame_bytes
        stream = process.stdout
        try:
            while True:
                chunk = stream.read1(_READ_CHUNK)
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
        if not self._stopping.is_set() and code not in (0, None):
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


class Player:
    """Queued playback of PCM clips, one sink subprocess per clip.

    One worker thread, started on the first ``play()`` so that constructing a
    Player touches no device.  ``stop()`` kills the running sink outright and
    drops what is queued: that is what makes barge-in feel immediate, because
    opening the microphone must silence the machine inside about 100 ms rather
    than at the end of the sentence it happens to be reading.
    """

    _JOIN_TIMEOUT = 2.0

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg if cfg is not None else {}
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._worker: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._generation = 0   # bumped by stop(); older clips are discarded
        self._pending = 0      # clips queued plus the one in flight
        self._closed = False
        self._error: str | None = None

    @property
    def playing(self) -> bool:
        """True while a clip is playing or waiting behind one."""
        with self._lock:
            return self._pending > 0

    @property
    def error(self) -> str | None:
        """Why the last clip failed, or None. A new play() clears it."""
        return self._error

    def play(self, pcm: bytes, rate: int) -> None:
        """Queue one s16le mono clip for playback and return immediately."""
        if self._closed:
            raise AudioError(
                "Player.play() was called after close(). Construct a new "
                "Player; a closed one has no worker left to play the clip.")
        if not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise AudioError(
                f"play() needs s16le PCM bytes, got {type(pcm).__name__}. "
                "Pass the bytes a TTS engine's synth() returned.")
        rate = _positive_int(rate, "the clip sample rate")
        data = bytes(pcm)
        # A half sample would shift the rest of the clip by one byte and turn
        # it into noise; the tail of an odd-length buffer goes.
        data = data[:len(data) - (len(data) % 2)]
        if not data:
            # Conditioning legitimately produces nothing to say (a pane of box
            # drawing, say); silence is the correct rendering of no audio.
            return
        with self._lock:
            self._error = None
            self._start_worker_locked()
            self._pending += 1
            self._idle.clear()
            self._queue.put((self._generation, data, rate))

    def stop(self) -> None:
        """Cancel playback now: drop the queue and kill the running sink."""
        with self._lock:
            self._generation += 1
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:      # close()'s sentinel is not a clip
                    self._queue.put(item)
                    break
                self._clip_done_locked()
            if self._process is not None:
                # SIGKILL, not SIGTERM: a sink asked politely may drain the
                # buffer it has already been handed, and keep talking over the
                # microphone we just opened.
                try:
                    self._process.kill()
                except OSError:
                    pass

    def wait(self, timeout: float | None = None) -> bool:
        """Block until nothing is playing; True when idle, False on timeout."""
        return self._idle.wait(timeout)

    def close(self) -> None:
        """Cancel playback and shut the worker down. Idempotent."""
        with self._lock:
            self._closed = True
            worker, self._worker = self._worker, None
        self.stop()
        if worker is not None:
            self._queue.put(None)
            worker.join(timeout=self._JOIN_TIMEOUT)

    def _start_worker_locked(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._run, name="kilix-voice-player", daemon=True)
            self._worker.start()

    def _clip_done_locked(self) -> None:
        self._pending -= 1
        if self._pending <= 0:
            self._pending = 0
            self._idle.set()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, pcm, rate = item
            try:
                self._play_clip(generation, pcm, rate)
            finally:
                with self._lock:
                    self._process = None
                    self._clip_done_locked()

    def _play_clip(self, generation: int, pcm: bytes, rate: int) -> None:
        """Run one sink process to completion, or until stop() kills it."""
        try:
            cmd = build_play_cmd(self._cfg, rate)
        except AudioError as error:
            # A machine with no sink installed must not take the worker down
            # with it: the failure is reported and the next clip still tries.
            self._error = str(error)
            return
        stderr = tempfile.TemporaryFile()
        try:
            with self._lock:
                if generation != self._generation:
                    return  # stopped while this clip waited in the queue
                try:
                    process = subprocess.Popen(
                        cmd, stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=stderr)
                except OSError as error:
                    self._error = (
                        f"cannot run the playback command {cmd[0]!r}: {error}. "
                        "Install that tool, or set audio.play_cmd to a command "
                        "that reads raw s16le mono audio from stdin.")
                    return
                self._process = process
            try:
                # The write blocks once the sink's buffer fills, which is the
                # flow control: the clip is handed over at the speed it plays,
                # so a kill from stop() lands mid-clip instead of after it.
                process.stdin.write(pcm)
            except (OSError, ValueError):
                pass  # killed by stop(), or the sink died mid-clip
            finally:
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
            code = process.wait()
            if code != 0 and generation == self._generation:
                self._error = _exit_message("playback", cmd, code, stderr,
                                            _PLAYBACK_HINT)
        finally:
            stderr.close()
