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

from . import resources


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
    """One immutable entry in the local speech-model catalog.

    C06 adds a device class and C12 a measured resource profile. Both are
    OPTIONAL with defaults, which is precisely what C01/V01 require: a legacy
    entry written before these existed stays constructible and valid. Adding
    them as required fields would have invalidated every legacy entry, which is
    the outcome V01 exists to prevent.
    """

    catalog_id: str
    engine: str
    size: int
    runtime_supported: bool
    summary: str
    device_class: str = resources.DEVICE_CPU
    resource_profile: dict | None = None


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


class CatalogError(ValueError):
    """A malformed speech-model catalog document."""


# The fields a v1 record is REQUIRED to carry. A reader that demanded the exact
# key set would break the moment a producer added an optional field -- which is
# precisely what C13 forbids -- so unknown keys are preserved and ignored, and
# only these are checked.
_RECORD_REQUIRED = ("id", "engine")
# C15: fields a consumer may ACT on, as opposed to inert metadata it merely
# carries forward. Preserving an unknown field is forward compatibility;
# handing back an unvalidated command vector is not. install_and_default_argv
# is emitted by the producer and is literally an argv.
_ACTIONABLE = ("install_and_default_argv",)
_ARGV_HEAD = "kilix"
_ARGV_OPTIONS = ("--install", "--default")
_RECORD_TYPES = {
    "id": str, "engine": str, "download_bytes": int, "download_size": str,
    "installed": bool, "runtime_supported": bool, "selected": bool,
    "path": str, "summary": str, "device_class": str,
}


def read_catalog(document: object) -> dict:
    """Return a validated speech-model catalog, ignoring unknown fields.

    The producer side has existed since the schema was introduced; nothing
    consumed it, so no caller could act on a catalog without re-implementing its
    shape. This is the reader.

    Forward compatibility is the point: an unknown top-level key or record field
    is carried through untouched, so a newer producer does not break an older
    reader. What is NOT tolerated is a wrong schema, a mistyped known field, or a
    duplicate id -- those change the meaning of fields a caller acts on.
    """
    if not isinstance(document, dict):
        raise CatalogError(
            f"a catalog document must be a JSON object, got "
            f"{type(document).__name__}.")
    schema = document.get("schema")
    if schema != CATALOG_SCHEMA:
        raise CatalogError(
            f"catalog schema is {schema!r}; this reader speaks "
            f"{CATALOG_SCHEMA!r}. A different schema may have changed what a "
            "field means.")
    records = document.get("models")
    if not isinstance(records, list):
        raise CatalogError(
            f"'models' must be a list, got {type(records).__name__}.")
    seen: set[str] = set()
    parsed = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise CatalogError(
                f"model record {index} must be an object, got "
                f"{type(record).__name__}.")
        for key in _RECORD_REQUIRED:
            if key not in record:
                raise CatalogError(
                    f"model record {index} is missing required field {key!r}.")
        for key, want in _RECORD_TYPES.items():
            if key in record and (not isinstance(record[key], want)
                                  or (want is int and isinstance(record[key], bool))):
                raise CatalogError(
                    f"model record {index} field {key!r} must be "
                    f"{want.__name__}, got {type(record[key]).__name__}.")
        if "device_class" in record and record["device_class"] not in resources.DEVICE_CLASSES:
            raise CatalogError(
                f"model record {index} device_class must be one of: "
                f"{', '.join(resources.DEVICE_CLASSES)}, got "
                f"{record['device_class']!r}.")
        if ("device_class" in record and "resource_profile" in record
                and isinstance(record["resource_profile"], dict)
                and record["resource_profile"].get("device_class")
                != record["device_class"]):
            # Two device declarations that disagree cannot both be acted on,
            # and picking one silently is how a cuda model gets scheduled onto
            # a cpu. Refuse the record instead.
            raise CatalogError(
                f"model record {index} declares device_class "
                f"{record['device_class']!r} but its resource_profile declares "
                f"{record['resource_profile'].get('device_class')!r}; they must "
                "agree.")
        if "resource_profile" in record:
            # C18: conformance is checked here, at the boundary, so a
            # malformed profile cannot reach a consumer that would size
            # hardware from it.
            try:
                resources.validate(record["resource_profile"])
            except resources.ResourceError as error:
                raise CatalogError(
                    f"model record {index} resource_profile is invalid: "
                    f"{error}") from error
        if "install_and_default_argv" in record:
            argv = record["install_and_default_argv"]
            if not isinstance(argv, list) or not argv:
                raise CatalogError(
                    f"model record {index} install_and_default_argv must be a "
                    f"non-empty list, got {type(argv).__name__}.")
            if any(not isinstance(a, str) for a in argv):
                raise CatalogError(
                    f"model record {index} install_and_default_argv must hold "
                    "only strings; a nested structure is not an argv.")
            if argv[0] != _ARGV_HEAD:
                raise CatalogError(
                    f"model record {index} install_and_default_argv starts "
                    f"with {argv[0]!r}; only {_ARGV_HEAD!r} is accepted. A "
                    "catalog may not name the program a consumer runs.")
            for a in argv:
                if a.startswith("-") and a not in _ARGV_OPTIONS:
                    raise CatalogError(
                        f"model record {index} install_and_default_argv "
                        f"carries option {a!r}; only "
                        f"{' and '.join(_ARGV_OPTIONS)} are accepted.")
                if "\x00" in a or "\n" in a:
                    raise CatalogError(
                        f"model record {index} install_and_default_argv holds "
                        "a control character.")
        catalog_id = record["id"]
        if catalog_id in seen:
            raise CatalogError(
                f"duplicate model id {catalog_id!r}; ids must be unique.")
        seen.add(catalog_id)
        parsed.append(dict(record))          # unknown fields preserved verbatim
    default = document.get("default_model")
    if default is not None and not isinstance(default, str):
        raise CatalogError(
            f"'default_model' must be a string or absent, got "
            f"{type(default).__name__}.")
    if default is not None and default not in seen:
        raise CatalogError(
            f"'default_model' is {default!r}, which is not one of the "
            f"{len(seen)} records in the document.")
    result = dict(document)
    result["models"] = parsed
    return result
