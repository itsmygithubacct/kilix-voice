"""Offline Whistle provider: a private process owns the native global model.

Only explicit local files are loaded. No upstream Python package, downloader,
telemetry or tool-calling model is involved. The PCM protocol is shared with
the Whisper provider. Short turns use one-shot decoding; longer turns use the
native streaming API so the 30-second one-shot limit cannot truncate speech.
"""
from __future__ import annotations

import argparse
import array
import ctypes
import json
import os
import sys
import wave

RATE = 16000
MAX_PCM_BYTES = RATE * 2 * 600
MAX_HEADER_BYTES = 256
MAX_MODEL_BYTES = 64 * 1024 * 1024
REPLY_BYTES = 1 << 18


class ProviderError(Exception):
    pass


class Whistle:
    def __init__(self, model: str, library: str, language: str = "en"):
        self.language = None if language == "auto" else language.encode("ascii")
        try:
            if not os.path.isfile(library):
                raise ProviderError(f"local Whistle library not found: {library}")
            path = os.path.join(model, "whistle.cact")
            with open(path, "rb") as source:
                weights = source.read(MAX_MODEL_BYTES + 1)
            if not weights or len(weights) > MAX_MODEL_BYTES:
                raise ProviderError("Whistle weights are empty or too large")
            lib = self.lib = ctypes.CDLL(os.path.abspath(library))
            samples = ctypes.POINTER(ctypes.c_float)
            lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
            lib.needle_load.restype = ctypes.c_int
            lib.needle_last_error.argtypes = []
            lib.needle_last_error.restype = ctypes.c_char_p
            lib.needle_transcribe.argtypes = [samples, ctypes.c_int, ctypes.c_char_p,
                                             ctypes.c_char_p, ctypes.c_int,
                                             ctypes.c_char_p, ctypes.c_int]
            lib.needle_transcribe.restype = ctypes.c_int
            lib.needle_stream_transcribe_process.argtypes = [
                samples, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_int]
            lib.needle_stream_transcribe_process.restype = ctypes.c_int
            lib.needle_stream_transcribe_stop.argtypes = [ctypes.c_char_p, ctypes.c_int]
            lib.needle_stream_transcribe_stop.restype = ctypes.c_int
            self.buffer = ctypes.create_string_buffer(REPLY_BYTES)
            self._check(lib.needle_load(weights, len(weights)))
        except (OSError, AttributeError) as error:
            raise ProviderError(f"cannot load local Whistle runtime: {error}") from error

    def _check(self, code):
        if code < 0:
            message = self.lib.needle_last_error()
            raise ProviderError((message or b"Whistle failed").decode("utf-8", "replace"))

    def _result(self, code):
        self._check(code)
        try:
            result = json.loads(self.buffer.value)
        except (ValueError, UnicodeError) as error:
            raise ProviderError("Whistle returned invalid JSON") from error
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ProviderError("Whistle returned no transcript")
        return result

    def transcribe(self, pcm: bytes) -> str:
        if len(pcm) % 2 or len(pcm) > MAX_PCM_BYTES:
            raise ProviderError("expected at most ten minutes of 16-bit PCM")
        if not pcm:
            return ""
        values = array.array("h", pcm)
        if sys.byteorder != "little":
            values.byteswap()
        floats = array.array("f", (value / 32768.0 for value in values))
        samples = (ctypes.c_float * len(floats)).from_buffer(floats)
        self.buffer.value = b""
        if len(samples) <= RATE * 30:
            return self._result(self.lib.needle_transcribe(
                samples, len(samples), self.language, None, 0,
                self.buffer, REPLY_BYTES))["text"]
        texts = []
        stopped = False
        try:
            for start in range(0, len(samples), RATE):
                count = min(RATE, len(samples) - start)
                ptr = ctypes.cast(ctypes.byref(samples, start * ctypes.sizeof(ctypes.c_float)),
                                  ctypes.POINTER(ctypes.c_float))
                self.buffer.value = b""
                texts.append(self._result(self.lib.needle_stream_transcribe_process(
                    ptr, count, self.language, None, self.buffer, REPLY_BYTES))["text"])
            self.buffer.value = b""
            code = self.lib.needle_stream_transcribe_stop(self.buffer, REPLY_BYTES)
            stopped = True
            texts.append(self._result(code)["text"])
        finally:
            if not stopped:
                self.lib.needle_stream_transcribe_stop(self.buffer, REPLY_BYTES)
        return " ".join(text.strip() for text in texts if text.strip())


def read_wav(path):
    try:
        with wave.open(path, "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
                raise ProviderError("audio must be 16 kHz mono 16-bit PCM")
            size = wav.getnframes() * 2
            if size > MAX_PCM_BYTES:
                raise ProviderError("audio exceeds ten minutes")
            pcm = wav.readframes(wav.getnframes())
            if len(pcm) != size:
                raise ProviderError("truncated WAV audio")
            return pcm
    except (OSError, EOFError, wave.Error) as error:
        raise ProviderError(f"cannot read audio: {error}") from error


def protocol_stdout():
    sys.stdout.flush()
    private = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(private, "wb", buffering=0)


def reply(out, value):
    out.write((json.dumps(value, ensure_ascii=True) + "\n").encode("ascii"))


def serve(args, stdin=None, out=None):
    stdin = stdin if stdin is not None else sys.stdin.buffer
    out = out if out is not None else protocol_stdout()
    try:
        model = Whistle(args.model, args.library, args.language)
    except ProviderError as error:
        reply(out, {"error": str(error)})
        return 1
    reply(out, {"ready": True, "version": "1"})
    while True:
        header = stdin.readline(MAX_HEADER_BYTES + 1)
        if not header:
            return 0
        try:
            if len(header) > MAX_HEADER_BYTES or not header.endswith(b"\n"):
                raise ProviderError("request header is not one short line")
            try:
                request = json.loads(header)
            except ValueError as error:
                raise ProviderError("request header is not JSON") from error
            size = request.get("pcm_bytes") if isinstance(request, dict) else None
            if (not isinstance(request, dict) or set(request) != {"pcm_bytes"}
                    or type(size) is not int or not 0 <= size <= MAX_PCM_BYTES or size % 2):
                raise ProviderError("expected {\"pcm_bytes\": N} with even N up to " + str(MAX_PCM_BYTES))
            chunks = []
            while size:
                chunk = stdin.read(size)
                if not chunk:
                    raise ProviderError("standard input ended inside a request")
                chunks.append(chunk)
                size -= len(chunk)
            pcm = b"".join(chunks)
        except ProviderError as error:
            reply(out, {"error": str(error)})
            return 2
        try:
            reply(out, {"text": model.transcribe(pcm)})
        except ProviderError as error:
            reply(out, {"error": str(error)})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline Cactus Whistle recognition")
    parser.add_argument("command", choices=("serve", "transcribe"))
    parser.add_argument("--model", required=True, help="local directory containing whistle.cact")
    parser.add_argument("--library", required=True, help="local libneedle3.so")
    parser.add_argument("--language", choices=("en", "de", "fr", "es", "it", "nl", "pl", "auto"), default="en")
    parser.add_argument("--audio", help="16 kHz mono 16-bit WAV (transcribe only)")
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve(args)
    if not args.audio:
        parser.error("transcribe requires --audio")
    out = protocol_stdout()
    try:
        pcm = read_wav(args.audio)
        text = Whistle(args.model, args.library, args.language).transcribe(pcm)
        out.write((text + "\n").encode("utf-8"))
        return 0
    except ProviderError as error:
        print(f"Whistle: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
