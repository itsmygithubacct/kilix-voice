"""Canonical local speech-model catalog for every Kilix Voice surface.

The catalog is deliberately data-only: importing it performs no filesystem,
subprocess, audio, or network work. The TUI, CLI, settings validator, and
machine-readable control-plane output all consume these same records. Other
processes should use ``kilix-stt --models --json`` instead of importing this
private package from another release closure.
"""

from __future__ import annotations

from typing import NamedTuple


CATALOG_SCHEMA = "kilix.speech.models/v1"

ENGINE_VOSK = "vosk"
ENGINE_VIBEVOICE = "vibevoice"
ENGINE_OFF = "off"
ENGINE_CHOICES = (ENGINE_VOSK, ENGINE_VIBEVOICE, ENGINE_OFF)


class ModelSpec(NamedTuple):
    """One immutable entry in the local speech-model catalog."""

    catalog_id: str
    engine: str
    size: int
    runtime_supported: bool
    summary: str


# Sizes are archive download sizes, not estimates of installed disk or working
# memory. Showing exact bytes lets every interface quote the cost before an
# explicit install without probing the network.
MODELS = (
    ModelSpec(
        "small-en-us",
        ENGINE_VOSK,
        41205931,
        True,
        "compact English model: the default, and the one that starts fastest "
        "on a laptop",
    ),
    ModelSpec(
        "lgraph-en-us",
        ENGINE_VOSK,
        130557655,
        True,
        "larger English graph: steadier on long dictation, slower to load "
        "and heavier on memory",
    ),
    ModelSpec(
        "vibevoice-asr-bitnet",
        ENGINE_VIBEVOICE,
        1705771590,
        False,
        "shared with Kilix Bonsai; its weights can be installed and selected "
        "here, but this voice runtime cannot run them yet",
    ),
)

MODEL_BY_ID = {spec.catalog_id: spec for spec in MODELS}
MODEL_IDS = tuple(MODEL_BY_ID)

# A published model directory is installed only when every file its engine
# needs exists and is non-empty. Installers publish these paths atomically; the
# check still refuses half-copied or manually damaged payloads.
REQUIRED_FILES = {
    ENGINE_VOSK: ("conf/model.conf", "am/final.mdl"),
    ENGINE_VIBEVOICE: (
        "vibeasr-lm-i2_s-embed-q6_k.gguf",
        "vibeasr-vae-encoder-i8_s.gguf",
    ),
}


def engine_for_model(catalog_id: str) -> str:
    """Return the matching recognizer, or raise ``KeyError`` for bad input."""
    return MODEL_BY_ID[catalog_id].engine


__all__ = [
    "CATALOG_SCHEMA",
    "ENGINE_CHOICES",
    "ENGINE_OFF",
    "ENGINE_VIBEVOICE",
    "ENGINE_VOSK",
    "MODELS",
    "MODEL_BY_ID",
    "MODEL_IDS",
    "ModelSpec",
    "REQUIRED_FILES",
    "engine_for_model",
]
