#!/usr/bin/env python3
"""Measure offline Pocket CPU load and submit-to-complete PCM on admitted weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicelib.pocket import ResidentPocket  # noqa: E402

DEFAULT_TEXT = "Good morning. This is a short speech test for the Kilix desktop."


def run(model_dir: Path, text: str, repeats: int, *, threads: int = 4) -> tuple[dict, bytes, int]:
    if not 1 <= repeats <= 20 or not text.strip() or len(text.encode("utf-8")) > 16384:
        raise ValueError("use 1–20 runs and 1–16384 UTF-8 bytes of text")
    result = {"schema": "kilix.tts.reply-benchmark/v1",
              "utc": datetime.now(timezone.utc).isoformat(),
              "model": "pocket-tts-english-cpu-alba", "text": text, "threads": threads,
              "runs": [], "measurement": "submit-to-complete-playable-PCM",
              "playback_included": False, "licence_or_install_action": False}
    start = time.monotonic()
    engine = ResidentPocket(model_dir, threads=threads)
    result["load_seconds"] = time.monotonic() - start
    try:
        for index in range(repeats):
            tick = time.monotonic()
            pcm, rate = engine.synth(text)
            elapsed = time.monotonic() - tick
            if not pcm or len(pcm) % 2 or rate != 24000:
                raise RuntimeError("Pocket returned invalid audio")
            duration = len(pcm) / (2 * rate)
            result["runs"].append({"index": index,
                                   "phase": "first" if index == 0 else "warm",
                                   "submit_to_pcm_seconds": elapsed,
                                   "audio_seconds": duration, "rtf": elapsed / duration,
                                   "pcm_sha256": hashlib.sha256(pcm).hexdigest()})
            if index == 0:
                result["load_and_first_reply_seconds"] = time.monotonic() - start
                first_pcm, first_rate = pcm, rate
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return result, first_pcm, first_rate
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="already admitted, local pinned Pocket model")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; never overwrite")
    args = parser.parse_args()
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be 1–32")
    if args.output.exists() or args.output.with_suffix(".wav").exists():
        parser.error("output JSON and WAV paths must not already exist")
    result, pcm, rate = run(args.model_dir, args.text, args.runs, threads=args.threads)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    with args.output.with_suffix(".wav").open("xb") as target:
        with wave.open(target, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(pcm)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
