"""Explicit offline Pocket TTS CPU audition with the pinned Alba preset only.

This is not a general voice-cloning interface or a release-qualified provider.
Weights and the preset embedding must already have passed first-use admission.
"""
from __future__ import annotations

import hashlib
import logging
from importlib.resources import files
import os
from pathlib import Path
import tempfile
import threading

from . import tts

PINNED = {
    "model.safetensors": (219029196, "be9c6b4876d3f30740a8225dfcaa2e43dc4aeb753c15272735bee16bbb4abb0a"),
    "tokenizer.model": (59339, "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"),
    "embeddings/alba.safetensors": (6194424, "e4a323e8e771f14fd669428d83e7af6d83b4d6d0f0e282c30122f04824f17b1c"),
}


def verify(directory: str | Path) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise tts.TtsError("--pocket-model-dir must be an existing local directory")
    result = {}
    for name, (size, digest) in PINNED.items():
        path = root / name
        if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
                or path.stat().st_size != size):
            raise tts.TtsError(f"Pocket model file missing or wrong size: {name}")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != digest:
            raise tts.TtsError(f"Pocket model file checksum mismatch: {name}")
        result[name] = path
    return result


STOP_SECONDS = 30


class _Stopped(Exception):
    """Raised inside Pocket's generator thread so its own error path ends the decoder."""


_STOP_MESSAGE = "kilix: reply interrupted"


def _hide_stop_report(record):
    # Upstream logs every generator exception at ERROR; an interrupt is not one.
    return _STOP_MESSAGE not in record.getMessage()


class ResidentPocket:
    """One warm CPU model; no network and no arbitrary voice/audio input."""

    def __init__(self, directory, *, voice=None, threads=4, seed=0):
        if voice and voice.lower() != "alba":
            raise tts.TtsError("Pocket audition offers only the licensed Alba preset")
        paths = verify(directory)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import numpy as np
            import torch
            import yaml
            from pocket_tts import TTSModel
        except ImportError as error:
            raise tts.TtsError("Pocket dependencies are missing; use the locked Pocket CPU runtime") from error
        torch.set_num_threads(threads)
        self.np, self.torch, self.seed = np, torch, seed
        source = files("pocket_tts").joinpath("config/english_2026-04.yaml")
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["weights_path"] = str(paths["model.safetensors"])
        config["weights_path_without_voice_cloning"] = str(paths["model.safetensors"])
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(paths["tokenizer.model"])
        with tempfile.TemporaryDirectory(prefix="kilix-pocket-config-") as scratch:
            local_config = Path(scratch) / "config.yaml"
            local_config.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.model = TTSModel.load_model(config=local_config)
        self.state = self.model.get_state_for_audio_prompt(paths["embeddings/alba.safetensors"])
        self._install_stop_hooks()
        self.name, self.voice = "Pocket TTS English CPU", "Alba"

    def voices(self):
        return ["Alba"]

    def set_voice(self, value):
        if value.lower() != "alba":
            raise tts.TtsError("only Alba is available in this Pocket audition")
        self.voice = "Alba"

    def _install_stop_hooks(self):
        """Let an interrupted reply stop Pocket's worker threads before returning.

        Upstream generation runs on daemon threads with no stop signal, so a
        Ctrl-C in the caller used to leave them decoding on the shared model for
        the rest of the reply. Each generation step now checks a stop flag and
        upstream's own error path then closes the decoder; every generator and
        decoder thread is counted so the caller waits for all of them, across
        every text chunk Pocket starts.
        """
        self._stop = threading.Event()
        self._workers = threading.Condition()
        self._live = 0
        self._stuck = False
        model = self.model
        hooks = ("_run_flow_lm_and_increment_step", "_decode_audio_worker", "_autoregressive_generation")
        if not all(callable(getattr(model, name, None)) for name in hooks):
            raise tts.TtsError("this Pocket runtime cannot stop an interrupted reply; use the locked runtime")
        step = model._run_flow_lm_and_increment_step

        def stoppable_step(*args, **kwargs):
            if self._stop.is_set():
                raise _Stopped(_STOP_MESSAGE)
            return step(*args, **kwargs)

        def counted(function):
            def worker(*args, **kwargs):
                with self._workers:
                    self._live += 1
                try:
                    return function(*args, **kwargs)
                finally:
                    with self._workers:
                        self._live -= 1
                        self._workers.notify_all()
            return worker

        model._run_flow_lm_and_increment_step = stoppable_step
        logging.getLogger("pocket_tts.models.tts_model").addFilter(_hide_stop_report)
        model._decode_audio_worker = counted(model._decode_audio_worker)
        model._autoregressive_generation = counted(model._autoregressive_generation)

    def _wait_for_workers(self):
        with self._workers:
            return self._workers.wait_for(lambda: self._live == 0, STOP_SECONDS)

    def synth(self, text):
        if not text.strip() or len(text.encode("utf-8")) > 16384:
            raise tts.TtsError("enter between 1 and 16384 UTF-8 bytes of text")
        if self._stuck:
            raise tts.TtsError("the interrupted Pocket reply did not stop; restart the session")
        self.torch.manual_seed(self.seed)
        self._stop.clear()
        # Pocket applies no_grad itself and hands mutable state to worker threads.
        # Inference mode is thread-local; its tensors cannot be updated there.
        try:
            wave = self.model.generate_audio(self.state, text)
        except BaseException:
            self._stop.set()
            if not self._wait_for_workers():
                self._stuck = True
            raise
        samples = wave.detach().cpu().numpy()
        if samples.ndim == 2 and samples.shape[0] == 1:
            samples = samples[0]
        rate = self.model.sample_rate
        if (rate != 24000 or samples.ndim != 1 or not samples.size
                or samples.size >= rate * 60 or not self.np.isfinite(samples).all()):
            raise tts.TtsError("Pocket returned invalid audio or reached the 60-second limit")
        return self.np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes(), rate

    def close(self):
        self.state = None
        self.model = None
