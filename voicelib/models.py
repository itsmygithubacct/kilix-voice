"""Canonical local speech-model catalogs for every Kilix Voice surface.

The catalog is deliberately data-only: importing it performs no filesystem,
subprocess, audio, or network work. The TUI, CLI, settings validator, and
machine-readable control-plane output all consume these same records. Other
processes should use ``kilix-stt --models --json`` for recognition artifacts
and ``kilix-tts --models`` for synthesis choices instead of importing this
private package from another release closure.
"""

from __future__ import annotations

from typing import NamedTuple


CATALOG_SCHEMA = "kilix.speech.models/v1"

ENGINE_VOSK = "vosk"
ENGINE_VIBEVOICE = "vibevoice"
ENGINE_OFF = "off"
ENGINE_CHOICES = (ENGINE_VOSK, ENGINE_VIBEVOICE, ENGINE_OFF)

TTS_ENGINE_ESPEAK = "espeak"
TTS_ENGINE_MBROLA = "mbrola"
TTS_ENGINE_PIPER = "piper"
PIPER_KRISTIN_MODEL = "piper-en-us-kristin-medium"


class ModelSpec(NamedTuple):
    """One immutable entry in the local speech-model catalog."""

    catalog_id: str
    engine: str
    size: int
    runtime_supported: bool
    summary: str


class TtsModelSpec(NamedTuple):
    """One synthesis family selectable by an untrusted speak request.

    The ID is deliberately separate from the engine name. They are equal for
    the two system synthesis families; the immutable Piper model ID maps to
    its isolated provider without widening the wire protocol to arbitrary
    executable names or model paths.
    """

    catalog_id: str
    engine: str
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

TTS_MODELS = (
    TtsModelSpec(
        "espeak",
        TTS_ENGINE_ESPEAK,
        True,
        "compact rule-based local speech with the installed eSpeak voices",
    ),
    TtsModelSpec(
        "mbrola",
        TTS_ENGINE_MBROLA,
        True,
        "local MBROLA diphone voices through eSpeak; explicit model requests "
        "fail closed when the voice is absent",
    ),
    TtsModelSpec(
        PIPER_KRISTIN_MODEL,
        TTS_ENGINE_PIPER,
        True,
        "local neural US English speech through the isolated persistent "
        "Piper provider and its pinned Kristin medium voice",
    ),
)

TTS_MODEL_BY_ID = {spec.catalog_id: spec for spec in TTS_MODELS}
TTS_MODEL_IDS = tuple(TTS_MODEL_BY_ID)

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


def tts_engine_for_model(catalog_id: str) -> str:
    """Return the matching synthesizer, or raise ``KeyError`` for a bad ID."""
    return TTS_MODEL_BY_ID[catalog_id].engine


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
    "TTS_ENGINE_ESPEAK",
    "TTS_ENGINE_MBROLA",
    "TTS_ENGINE_PIPER",
    "TTS_MODELS",
    "TTS_MODEL_BY_ID",
    "TTS_MODEL_IDS",
    "PIPER_KRISTIN_MODEL",
    "TtsModelSpec",
    "engine_for_model",
    "tts_engine_for_model",
]
