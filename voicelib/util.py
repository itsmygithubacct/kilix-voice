"""Small shared helpers used by every voicelib module.

Frozen contract — see DESIGN.md.  Audio here is always signed 16-bit
little-endian mono PCM; anything else is a caller error, not something this
module quietly converts.
"""

from __future__ import annotations

import array
import io
import math
import pathlib
import shutil
import struct
import sys
import wave

_WAVE_FORMAT_PCM = 1


def repo_root() -> pathlib.Path:
    """Return the checkout root (the directory holding VERSION and Makefile)."""
    return pathlib.Path(__file__).resolve().parent.parent


def which(name: str) -> str | None:
    """Return the absolute path of an executable on PATH, or None."""
    return shutil.which(name)


def cfg_get(cfg: dict, path: str, default=None):
    """Dotted lookup into a nested dict: cfg_get(cfg, "audio.frame_ms")."""
    current = cfg
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def rms16(frame: bytes) -> float:
    """Return the normalised RMS (0.0-1.0) of an s16le mono frame.

    Normalised rather than raw counts so the VAD's thresholds are expressed in
    the same units regardless of sample width.  An empty or odd-length tail
    yields 0.0 instead of raising: a truncated final read is normal.
    """
    usable = len(frame) - (len(frame) % 2)
    if usable <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frame[:usable])
    if sys.byteorder == "big":
        samples.byteswap()
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples)) / 32768.0


def parse_wav_bytes(data: bytes) -> tuple[bytes, int]:
    """Return (s16le mono PCM, sample rate) from an in-memory WAV.

    Walks the RIFF chunk list instead of assuming header offsets: real engine
    output carries optional chunks ('fact', 'LIST') before the data, and a
    piped writer reports a placeholder data size that is nothing like the
    number of bytes it went on to write.  Raises ValueError for anything that
    is not uncompressed 16-bit mono PCM — resampling and downmixing belong to
    the caller that knows what the audio is for.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(
            "not a WAV stream: missing the RIFF/WAVE header. Check that the "
            "engine was asked for WAV on stdout and that stderr was not mixed "
            "into the same pipe.")
    fmt: tuple[int, ...] | None = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > len(data):
                raise ValueError(
                    "truncated WAV 'fmt ' chunk: the stream ended mid-header. "
                    "Read the engine's output to completion before parsing.")
            fmt = struct.unpack_from("<HHIIHH", data, body)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError(
                    "malformed WAV: the 'data' chunk precedes 'fmt '. Re-run "
                    "the engine; this stream cannot be interpreted.")
            tag, channels, rate, _byte_rate, _align, bits = fmt
            if tag != _WAVE_FORMAT_PCM:
                raise ValueError(
                    f"unsupported WAV format tag {tag}: kilix-voice needs "
                    "uncompressed PCM (tag 1). Ask the engine for raw PCM/WAV "
                    "output rather than a compressed container.")
            if channels != 1:
                raise ValueError(
                    f"WAV declares {channels} channels: kilix-voice audio is "
                    "mono. Configure the engine for 1 channel.")
            if bits != 16:
                raise ValueError(
                    f"WAV declares {bits}-bit samples: kilix-voice audio is "
                    "signed 16-bit. Configure the engine for 16-bit output.")
            if rate <= 0:
                raise ValueError(
                    "WAV header declares a sample rate of 0. Re-run the "
                    "engine; the stream is corrupt.")
            # A writer piping WAV cannot seek back to patch its header, so the
            # declared length is a placeholder: espeak leaves 0x7FFFFF80,
            # others leave 0 or ~0u. Trust the bytes that actually arrived.
            end = body + size if 0 < size <= len(data) - body else len(data)
            pcm = data[body:end]
            return pcm[:len(pcm) - (len(pcm) % 2)], rate
        # Chunks are word-aligned: an odd size is followed by a pad byte.
        pos = body + size + (size % 2)
    raise ValueError(
        "WAV stream has no 'data' chunk: nothing was synthesised. Check the "
        "engine's exit status and stderr.")


def write_wav(pcm: bytes, rate: int) -> bytes:
    """Wrap s16le mono PCM in a WAV container and return the bytes."""
    if rate <= 0:
        raise ValueError(
            f"sample rate must be positive, got {rate!r}. Pass the rate the "
            "engine reported for this clip.")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        # A half sample would desynchronise every following frame; drop it.
        out.writeframes(pcm[:len(pcm) - (len(pcm) % 2)])
    return buffer.getvalue()
