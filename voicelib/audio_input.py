"""Caller audio, checked against what the caller declared (A01, A06, A08).

A caller hands the daemon audio as a descriptor with typed declarations. This
module turns the bytes read from it into the one shape the daemon works in --
s16le mono PCM at a known rate -- or refuses them with a closed code. Refuse,
never trim: a clip cut to fit a limit would be a wrong answer wearing a
success, the same rule A06 applies everywhere else. Nothing is resampled or
downmixed. Normalisation here means canonicalising the container and checking
the declarations.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from . import protocol, util


class AudioInputError(ValueError):
    """Audio a caller handed over that cannot be used; ``code`` says why."""

    def __init__(self, message: str, code: str = protocol.ERR_MALFORMED) -> None:
        if code not in protocol.ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code


class NormalizedAudio(NamedTuple):
    pcm: bytes
    sample_rate: int
    duration_ms: int


def normalize(data: bytes, declared: dict) -> NormalizedAudio:
    """Return the PCM in ``data`` checked against ``declared``, or refuse it.

    ``declared`` is the normalised request: container, sample_rate and
    duration_limit_ms are read when present. A WAV carries its own rate, and a
    declared rate must agree with it. Raw PCM has no header, so its rate must
    be declared.
    """
    container = declared.get("container") or (
        "wav" if data[:4] == b"RIFF" else "raw")
    rate = declared.get("sample_rate")
    if container == "wav":
        try:
            pcm, header_rate = util.parse_wav_bytes(data, strict=True)
        except util.WavFormatError as error:
            raise AudioInputError(str(error), error.code) from error
        if rate is not None and rate != header_rate:
            raise AudioInputError(
                f"the WAV header says {header_rate} Hz and the request declared "
                f"sample_rate {rate}. Declare the rate the audio has, or omit "
                "sample_rate for WAV input.")
        rate = header_rate
    else:
        if rate is None:
            raise AudioInputError(
                "raw PCM needs a declared sample_rate: nothing in the bytes "
                "says what it is. Add sample_rate, or send a WAV.")
        if len(data) % 2:
            raise AudioInputError(
                f"raw s16le PCM is whole 2-byte samples, and {len(data)} bytes "
                "is not. Send the samples as they were written.")
        pcm = bytes(data)
    if not protocol.MIN_SAMPLE_RATE <= rate <= protocol.MAX_SAMPLE_RATE:
        raise AudioInputError(
            f"a sample rate of {rate} Hz is not supported; use "
            f"{protocol.MIN_SAMPLE_RATE}-{protocol.MAX_SAMPLE_RATE} Hz. Audio is "
            "not resampled.", protocol.ERR_UNSUPPORTED)
    duration_ms = math.ceil(len(pcm) * 1000 / (2 * rate))
    limit = declared.get("duration_limit_ms", protocol.MAX_INGEST_DURATION_MS)
    if duration_ms > limit:
        raise AudioInputError(
            f"the audio lasts {duration_ms} ms and the duration limit is {limit} "
            "ms. It was refused, not truncated.", protocol.ERR_TOO_LARGE)
    return NormalizedAudio(pcm, rate, duration_ms)
