"""Foreground, local TTS sessions. No daemon settings or history are written.

The optional resident Qwen runtime is an explicit development audition path,
not the receipt-checked provider/install path. It only opens existing local
weights, never downloads them, and does not claim release qualification.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from . import audio, tts, util

HELP = """Type a line and press Enter to generate and play it on this machine.
/help              show this help
/voices            list available voices
/voice NAME        select a voice (CustomVoice or conventional engines)
/language NAME     change Qwen language, e.g. English, Japanese, Auto
/repeat            replay the last completed clip
/save FILE.wav     save the last clip; never overwrite a file
/quit              exit and release the model
Ctrl-C cancels the current operation; Ctrl-D exits. This tool saves no prompt history.
"""
REFERENCE_TEXT = "Hello. This is a synthetic reference voice for a speech test."
MAX_TEXT_BYTES = 16384


class ResidentQwen:
    """One warm, offline upstream model owned by the foreground session."""

    def __init__(self, directory, *, voice=None, language="English",
                 device="cpu", threads=4, seed=0, synthetic_reference=False,
                 attention="sdpa"):
        if attention not in {"sdpa", "flash_attention_2"}:
            raise tts.TtsError("unsupported attention backend")
        if attention == "flash_attention_2" and device == "cpu":
            raise tts.TtsError("FlashAttention requires --device cuda:0")
        root = Path(directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise tts.TtsError("--qwen-model-dir must be an existing local directory")
        config = json.loads((root / "config.json").read_text())
        if not isinstance(config, dict):
            raise tts.TtsError("Qwen config.json must contain an object")
        self.kind = config.get("tts_model_type")
        if self.kind not in {"base", "custom_voice"}:
            raise tts.TtsError("this session supports Qwen Base and CustomVoice models")
        if self.kind == "base" and (not synthetic_reference or voice):
            raise tts.TtsError("Base requires --synthetic-reference and no --voice; "
                               "this audition does not clone people's voices")
        if self.kind != "base" and synthetic_reference:
            raise tts.TtsError("--synthetic-reference requires a Base model")
        self.language, self.seed = language, seed
        self.voice = voice or ("synthetic-espeak" if self.kind == "base" else "Ryan")
        self.name = root.name
        self.prompt = None
        # Both the model and its nested speech tokenizer must remain offline.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import numpy as np
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as error:
            raise tts.TtsError("Qwen dependencies are missing. Run kilix-tts with "
                               "the Python from your isolated qwen-tts environment") from error
        self.np, self.torch = np, torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
        if device != "cpu" and not torch.cuda.is_available():
            raise tts.TtsError("CUDA was requested but is unavailable; use --device cpu")
        if attention == "flash_attention_2":
            if torch.cuda.get_device_capability(device)[0] < 8:
                raise tts.TtsError("FlashAttention 2 requires an Ampere or newer NVIDIA GPU")
            try:
                import flash_attn  # noqa: F401 — load the extension before loading weights
            except (ImportError, OSError) as error:
                raise tts.TtsError("FlashAttention 2 could not load; install a flash-attn build "
                                   "matching this environment's CUDA PyTorch and Python") from error
        self.device, self.attention = device, attention
        self.model = Qwen3TTSModel.from_pretrained(
            str(root), device_map=device,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            attn_implementation=attention, local_files_only=True,
            use_safetensors=True, trust_remote_code=False)
        self.set_language(language)
        if self.kind == "base":
            binary = util.which("espeak-ng") or util.which("espeak")
            if not binary:
                raise tts.TtsError("the synthetic Base demo requires espeak-ng")
            with tempfile.TemporaryDirectory(prefix="kilix-tts-reference-") as tmp:
                filename = str(Path(tmp) / "reference.wav")
                subprocess.run([binary, "-v", "en-us", "-s", "150", "-w",
                                filename, REFERENCE_TEXT], check=True, timeout=15,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                pcm, rate = util.parse_wav_bytes(Path(filename).read_bytes())
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
                self.prompt = self.model.create_voice_clone_prompt(
                    ref_audio=(samples, rate), ref_text=REFERENCE_TEXT,
                    x_vector_only_mode=False)
        else:
            self.set_voice(self.voice)

    def voices(self):
        return (list(self.model.get_supported_speakers()) if self.kind == "custom_voice"
                else ["synthetic-espeak (fixed synthetic reference)"])

    def set_voice(self, value):
        if self.kind != "custom_voice":
            raise tts.TtsError("Base uses a fixed synthetic reference; use CustomVoice for named voices")
        choices = {v.lower(): v for v in self.voices()}
        if value.lower() not in choices:
            raise tts.TtsError("unknown voice; use /voices")
        self.voice = choices[value.lower()]

    def set_language(self, value):
        choices = {v.lower(): v for v in self.model.get_supported_languages()}
        if value.lower() not in choices:
            raise tts.TtsError("unsupported language; choose " + ", ".join(choices.values()))
        self.language = choices[value.lower()]

    def synth(self, text):
        if not text.strip() or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise tts.TtsError("enter between 1 and 16384 UTF-8 bytes of text")
        self.torch.manual_seed(self.seed)
        kwargs = dict(text=text, language=self.language,
                      non_streaming_mode=True, max_new_tokens=750)
        with self.torch.inference_mode():
            if self.kind == "base":
                waves, rate = self.model.generate_voice_clone(
                    **kwargs, voice_clone_prompt=self.prompt)
            else:
                waves, rate = self.model.generate_custom_voice(**kwargs, speaker=self.voice)
        if len(waves) != 1 or rate != 24000:
            raise tts.TtsError("Qwen returned an unexpected audio format")
        samples = self.np.asarray(waves[0])
        if (samples.ndim != 1 or not samples.size or samples.size >= rate * 60
                or not self.np.isfinite(samples).all()):
            raise tts.TtsError("Qwen returned invalid audio or reached the 60-second limit")
        return (self.np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes(), rate)

    def close(self):
        self.prompt = None
        self.model = None


def save_clip(filename, clip):
    destination = Path(filename).expanduser()
    if destination.suffix.lower() != ".wav":
        raise tts.TtsError("/save requires a .wav filename")
    pcm, rate = clip
    # Exclusive creation: existing files and symlinks are never replaced.
    with destination.open("xb") as output:
        import wave
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm)


def run(args, *, read=input, emit=print, engine_factory=None, player_factory=None):
    engine = player = None
    last = None
    try:
        emit("Kilix TTS interactive — foreground playback; shared settings unchanged.")
        if getattr(args, "pocket_model_dir", None):
            emit("Loading verified local Pocket TTS on CPU; first load may take a while…")
            from .pocket import ResidentPocket
            factory = engine_factory or ResidentPocket
            engine = factory(args.pocket_model_dir, voice=args.voice,
                             threads=args.threads, seed=args.seed)
        elif args.qwen_model_dir:
            emit(f"Loading local Qwen on {args.device} with {args.attention}; first load may take a while…")
            if args.synthetic_reference:
                emit("Base audition: synthetic eSpeak reference, not a human voice or a preset voice.")
            factory = engine_factory or ResidentQwen
            engine = factory(args.qwen_model_dir, voice=args.voice,
                             language=args.language or "English", device=args.device,
                             threads=args.threads, seed=args.seed,
                             synthetic_reference=args.synthetic_reference,
                             attention=args.attention)
        else:
            factory = engine_factory or tts.make_tts
            engine = factory(model=args.model, voice=args.voice, rate=args.rate)
        player = (player_factory or audio.Player)({})
        emit(HELP)
        emit(f"Ready: {engine.name} / {engine.voice}")
        pending = args.speak
        while True:
            try:
                line = pending if pending is not None else read("tts> ")
                pending = None
                line = line.strip()
                if not line:
                    continue
                if line in {"/quit", "/exit"}:
                    break
                if line == "/help":
                    emit(HELP)
                    continue
                if line == "/voices":
                    emit(", ".join(engine.voices()) if hasattr(engine, "voices")
                         else "Use the engine's voice names, e.g. en-us, en-gb.")
                    continue
                if line.startswith("/voice "):
                    value = line[7:].strip()
                    if args.qwen_model_dir or getattr(args, "pocket_model_dir", None):
                        engine.set_voice(value)
                    else:
                        replacement = factory(model=args.model, voice=value, rate=args.rate)
                        engine.close()
                        engine = replacement
                    emit(f"Voice: {engine.voice}")
                    continue
                if line.startswith("/language "):
                    if not args.qwen_model_dir:
                        raise tts.TtsError("/language is for Qwen; use /voice for this engine")
                    engine.set_language(line[10:].strip())
                    emit(f"Language: {engine.language}")
                    continue
                if line.startswith("/save "):
                    if last is None:
                        raise tts.TtsError("no completed clip to save yet")
                    save_clip(line[6:].strip(), last)
                    emit("Saved WAV.")
                    continue
                if line == "/repeat":
                    if last is None:
                        raise tts.TtsError("no completed clip to replay yet")
                elif line.startswith("/"):
                    raise tts.TtsError("unknown command; use /help")
                else:
                    if len(line.encode("utf-8")) > MAX_TEXT_BYTES:
                        raise tts.TtsError("text exceeds 16384 UTF-8 bytes")
                    started = time.monotonic()
                    emit("Generating…")
                    clip = engine.synth(line)
                    if not clip[0]:
                        raise tts.TtsError("the engine returned no audio")
                    last = clip
                    emit(f"Generated {len(clip[0]) / (2 * clip[1]):.2f}s audio "
                         f"in {time.monotonic() - started:.2f}s; playing…")
                player.play(*last)
                while not player.wait(timeout=0.1):
                    pass
                if player.error:
                    raise audio.AudioError(player.error)
                emit("Playback complete.")
            except EOFError:
                break
            except KeyboardInterrupt:
                player.stop()
                emit("\nCancelled. Enter another prompt, or /quit.")
            except (tts.TtsError, audio.AudioError, OSError, ValueError, RuntimeError) as error:
                emit(f"Error: {error}")
        return 0
    finally:
        if player is not None:
            player.close()
        if engine is not None:
            engine.close()
