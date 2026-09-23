#!/usr/bin/env python3
"""Measure prompt submission to complete, playable Piper PCM through Kilix.

Starts an owned provider in a private session namespace, then measures one
cold request and repeated warm requests through voicelib.tts.PiperTts. Never
downloads weights, changes settings, or touches another provider socket.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicelib import tts  # noqa: E402

DEFAULT_TEXT = "Good morning. This is a short speech test for the Kilix desktop."


def run(text: str, repeats: int) -> tuple[dict, bytes, int]:
    if not 1 <= repeats <= 20 or not text.strip() or len(text.encode()) > 16384:
        raise ValueError("use 1–20 runs and 1–16384 UTF-8 bytes of text")
    binary = tts.piper_binary()
    ready, detail = tts.piper_status()
    if not binary or not ready:
        raise RuntimeError(f"Piper model/provider is unavailable: {detail}")
    result = {"schema": "kilix.tts.reply-benchmark/v1", "utc": datetime.now(timezone.utc).isoformat(),
              "model": "piper-en-us-kristin-medium", "rate_wpm": 170,
              "text": text, "runs": [], "measurement": "submit-to-complete-playable-PCM",
              "playback_included": False, "licence_or_install_action": False}
    with tempfile.TemporaryDirectory(prefix="kilix-tts-reply-") as session:
        environment = dict(os.environ, KILIX_SESSION_HOME=session)
        previous = os.environ.get("KILIX_SESSION_HOME")
        os.environ["KILIX_SESSION_HOME"] = session
        provider = None
        engine = None
        try:
            start = time.monotonic()
            provider = subprocess.Popen([binary, "serve", "--idle-seconds", "30"],
                                        env=environment, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                        start_new_session=True)
            engine = tts.PiperTts(rate=170)
            # Wait for the owned listener. Otherwise the first client may see
            # ENOENT and launch a second provider in the same namespace.
            socket_path = Path(session) / "voice" / "piper.sock"
            deadline = time.monotonic() + 8
            while not socket_path.is_socket():
                if provider.poll() is not None:
                    raise RuntimeError("private Piper provider exited before its socket was ready")
                if time.monotonic() >= deadline:
                    raise RuntimeError("private Piper provider did not create its socket")
                time.sleep(.01)
            for index in range(repeats):
                tick = time.monotonic()
                pcm, rate = engine.synth(text)
                elapsed = time.monotonic() - tick
                if not pcm or len(pcm) % 2 or rate != 22050:
                    raise RuntimeError("Piper returned invalid audio")
                duration = len(pcm) / (2 * rate)
                result["runs"].append({"index": index, "phase": "cold" if index == 0 else "warm",
                                       "submit_to_pcm_seconds": elapsed,
                                       "audio_seconds": duration, "rtf": elapsed / duration,
                                       "pcm_sha256": hashlib.sha256(pcm).hexdigest()})
                if index == 0:
                    result["provider_start_and_first_reply_seconds"] = time.monotonic() - start
                    first_pcm, first_rate = pcm, rate
            return result, first_pcm, first_rate
        finally:
            if engine is not None:
                engine.close()
            if provider is not None:
                # The service and worker belong to this private process group.
                try:
                    os.killpg(provider.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    provider.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(provider.pid, signal.SIGKILL)
                    provider.wait()
                provider.stderr.close()
            if previous is None:
                os.environ.pop("KILIX_SESSION_HOME", None)
            else:
                os.environ["KILIX_SESSION_HOME"] = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; never overwrite")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".wav").exists():
        parser.error("output JSON and WAV paths must not already exist")
    result, pcm, rate = run(args.text, args.runs)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    wav = args.output.with_suffix(".wav")
    with wav.open("xb") as target:
        with wave.open(target, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(pcm)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
