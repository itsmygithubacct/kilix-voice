"""Explicit audition tiers; memory arithmetic belongs to the shared sizer.

Listing never imports model weights, acquires licences, starts a provider or
changes settings. Readiness is provisional: the engine still validates on load.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

from . import paths, pocket, sizing, tts, util

QWEN_ID = "qwen3-tts-0.6b-customvoice"
QWEN_BASE_ID = "qwen3-tts-0.6b-base"
POCKET_ID = "pocket-tts-english-python-alba"
TIERS = (
    ("minimal", "eSpeak", "espeak", "cpu"),
    ("small", "MBROLA us1", "mbrola", "cpu"),
    ("neural", "Piper Kristin medium", "piper-en-us-kristin-medium", "cpu"),
    ("pocket-cpu", "Pocket TTS English / Alba", POCKET_ID + "-cpu", "cpu"),
    ("qwen-cpu", "Qwen 0.6B CustomVoice", QWEN_ID + "-cpu", "cpu"),
    ("qwen-gpu", "Qwen 0.6B CustomVoice / FlashAttention 2", QWEN_ID + "-cuda", "cuda"),
    ("qwen-base-gpu", "Qwen 0.6B Base / synthetic reference", QWEN_BASE_ID + "-cuda", "cuda"),
)
TIER_IDS = tuple(row[0] for row in TIERS)


def qwen_directory(model_id: str = QWEN_ID) -> Path:
    if model_id not in (QWEN_ID, QWEN_BASE_ID):
        raise ValueError("unknown Qwen audition model")
    return Path(paths.gpu_terminal_home()) / "tts-auditions/content/assets" / model_id / "model"


def pocket_directory() -> Path:
    return Path(paths.gpu_terminal_home()) / "tts-auditions/content/assets" / POCKET_ID / "model"


def pocket_installed() -> bool:
    root = pocket_directory()
    try:
        return all(not (root / name).is_symlink() and (root / name).is_file()
                   and (root / name).stat().st_size == size
                   for name, (size, _) in pocket.PINNED.items())
    except OSError:
        return False


def pocket_runtime() -> bool:
    return all(importlib.util.find_spec(name) is not None
               for name in ("pocket_tts", "torch", "torchaudio", "numpy", "yaml"))


def qwen_installed(model_id: str = QWEN_ID) -> bool:
    root = qwen_directory() if model_id == QWEN_ID else qwen_directory(model_id)
    try:
        if (root / "config.json").stat().st_size > 1024 * 1024:
            return False
        config = json.loads((root / "config.json").read_text())
        kind = "base" if model_id == QWEN_BASE_ID else "custom_voice"
        return (isinstance(config, dict) and config.get("tts_model_type") == kind
                and (root / "model.safetensors").is_file()
                and (root / "speech_tokenizer/model.safetensors").is_file())
    except (OSError, ValueError):
        return False


def qwen_runtime() -> tuple[bool, bool]:
    """Probe this interpreter, not another venv. No weights are opened.

    The audition GPU selector is deliberately cuda:0 on the unmasked physical
    GPU 0. Refuse remapping rather than compare one GPU's RAM with another's.
    """
    try:
        if any(importlib.util.find_spec(name) is None for name in ("torch", "qwen_tts", "numpy")):
            return False, False
        raw = sizing._run([sys.executable, "-c", "import json, torch\n"
            "ok = False\n"
            "try:\n"
            " import flash_attn\n"
            " ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8\n"
            "except (ImportError, OSError, RuntimeError): pass\n"
            "print(json.dumps([True, bool(ok)]))"], b"")
        result = sizing._parse(raw)
        if not isinstance(result, list) or len(result) != 2 or any(type(v) is not bool for v in result):
            return False, False
        return result[0], result[1] and os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "0"
    except (ImportError, ValueError, sizing.SizerError):
        return False, False


def availability() -> dict:
    espeak = bool(util.which("espeak-ng") or util.which("espeak"))
    mbrola = bool(espeak and util.which("mbrola") and tts.mbrola_database_installed("us1"))
    piper, detail = tts.piper_status(budget=3)
    qwen, gpu = qwen_runtime()
    weights = qwen_installed()
    base_weights = qwen_installed(QWEN_BASE_ID)
    pocket_weights = pocket_installed()
    pocket_ready = pocket_runtime()
    return {
        "minimal": (espeak, espeak, "install espeak-ng"),
        "small": (mbrola, mbrola, "install espeak-ng, mbrola and the us1 voice"),
        "neural": (piper, bool(tts.piper_binary()), detail),
        "pocket-cpu": (pocket_weights, pocket_ready,
                       "needs admitted Pocket Alba weights and this Python's locked Pocket CPU runtime"),
        "qwen-cpu": (weights, qwen, "needs installed 0.6B CustomVoice weights and this Python's qwen-tts runtime"),
        "qwen-gpu": (weights, gpu, "needs installed weights, CUDA PyTorch, working FlashAttention 2 and an unmasked Ampere+ GPU 0"),
        "qwen-base-gpu": (base_weights, gpu and espeak,
                          "needs installed 0.6B Base weights, eSpeak, CUDA PyTorch, FlashAttention 2 and an unmasked Ampere+ GPU 0"),
    }


def report() -> dict:
    available = availability()
    request = {"schema": sizing.REQUEST_SCHEMA, "models": [
        {"id": model if tier == "qwen-base-gpu" else "audition-" + model,
         "task": "tts", "backend": backend,
         "installed": available[tier][0],
         # Qwen runtimes can be installed after a hardware-only fit check.
         # The installed dependency probes below still gate selection and
         # first-use weight acquisition; the GPU installer also checks Ampere+.
         "runtime_supported": True if tier in ("pocket-cpu", "qwen-cpu", "qwen-gpu", "qwen-base-gpu")
         else available[tier][1]}
        for tier, _, model, backend in TIERS]}
    result = sizing.recommend_request(request, "tts")
    candidates = {row["id"]: row for row in result["candidates"]}
    for tier, label, model, backend in TIERS:
        row = candidates[model if tier == "qwen-base-gpu" else "audition-" + model]
        installed, runtime, detail = available[tier]
        correct_gpu = backend != "cuda" or row.get("budget", {}).get("gpu_index") == 0
        selectable = installed and runtime and correct_gpu and row["verdict"] == "estimated-fit"
        installable = tier in ("neural", "pocket-cpu", "qwen-cpu", "qwen-gpu", "qwen-base-gpu") \
            and not installed and runtime and correct_gpu and row["verdict"] == "estimated-fit"
        reason = ("ready (reference-workload estimate)" if selectable else detail if not installed or not runtime
                  else "sizer assessed a different GPU; only physical GPU 0 is supported" if not correct_gpu
                  else "resource estimate: " + row["verdict"])
        if installable:
            reason = "available for first-use installation (licence acceptance required)"
        row.update(tier=tier, label=label, selectable=selectable,
                   installable=installable, availability_detail=reason)
    result["notes"].append("Audition tiers do not change daemon defaults or authorize installation. Use one at a time.")
    return result


def select(args, result: dict) -> None:
    row = next(row for row in result["candidates"] if row["tier"] == args.tier)
    if row["verdict"] != "estimated-fit" or (not row["selectable"] and not row["installable"]):
        raise sizing.SizerError(f"Tier {args.tier} unavailable: {row['availability_detail']}")
    _, _, model, backend = next(tier for tier in TIERS if tier[0] == args.tier)
    if args.tier == "pocket-cpu":
        if (args.voice and args.voice.lower() != "alba") or getattr(args, "rate", None) is not None or getattr(args, "language", None):
            raise sizing.SizerError("Pocket CPU offers only Alba at its natural rate; omit --rate and --language")
        args.pocket_model_dir = str(pocket_directory())
        args.voice = "Alba"
    elif args.tier.startswith("qwen-"):
        if args.tier == "qwen-base-gpu" and args.voice is not None:
            raise sizing.SizerError("Qwen Base uses a fixed synthetic reference; omit --voice")
        args.qwen_model_dir = str(qwen_directory(QWEN_BASE_ID) if args.tier == "qwen-base-gpu"
                                 else qwen_directory())
        args.device = "cpu" if backend == "cpu" else "cuda:0"
        args.attention = "sdpa" if backend == "cpu" else "flash_attention_2"
        if args.tier == "qwen-base-gpu":
            args.synthetic_reference = True
    else:
        args.model = model
        if args.tier == "small" and args.voice is None:
            args.voice = "us1"


def print_report(result: dict, as_json=False) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print("TTS audition tiers — explicit choice, no model downloads or settings changes.")
    for row in result["candidates"]:
        resources = (row.get("inference") or {}).get("resources", {})
        amounts = ", ".join(f"{name.upper()} {resources[name]['required_bytes'] / 1024**2:.0f} MiB"
                            for name in ("ram", "vram") if name in resources)
        print(f"{row['tier']:10} {row['label']}: {row['availability_detail']}"
              + (f" [{amounts}]" if amounts else ""))
    print("Run: kilix-tts --interactive --tier NAME")
    print("Figures include the profile margin; the sizer also keeps 256 MiB free in each memory pool.")
    print("Fit is a memory estimate, not a speed/quality guarantee; Qwen CPU can be very slow.")
