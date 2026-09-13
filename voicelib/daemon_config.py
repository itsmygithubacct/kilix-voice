"""The voice daemon's effective configuration, built in exactly one place.

kilix-voiced layers an optional JSON file (``--config`` or KILIX_VOICE_CONFIG)
over the shared Kilix settings. That result decides, among everything else,
which model directory dictation opens -- and so which bytes the consent gate
hashes. R6 finding 7: ``kilix-stt --grant-consent`` built its own idea of the
configuration, hashing the catalogue directory, and under a model override the
only command that records consent could never record one the gate accepts.
Both now build the configuration here.

Importing this module performs no filesystem work.
"""

from __future__ import annotations

import json

from . import audio, settings

ENV_CONFIG = "KILIX_VOICE_CONFIG"

# Nothing in flight for this long and the daemon exits: it is started on demand
# by the terminal, and a voice session nobody is using should not keep a
# process -- or an audio device -- reserved.
IDLE_SECONDS = 300.0


class ConfigError(RuntimeError):
    """The layered config file cannot be used; the message says what to do."""


def load_overrides(path: str) -> dict:
    """Return the JSON config layered over the shared settings."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    except OSError as error:
        raise ConfigError(
            f"cannot read the daemon config {path}: {error}. Point --config "
            f"(or {ENV_CONFIG}) at a readable JSON file, or drop it to use the "
            "shared Kilix settings alone.") from error
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise ConfigError(
            f"{path} is not valid JSON: {error}. It must hold a single "
            'object, for example {"audio": {"play_cmd": ["cat"]}}.') from error
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a JSON object, got {type(data).__name__}. "
            'Wrap the settings, for example {"daemon": {"idle_seconds": 30}}.')
    return data


def merge(base: dict, extra: dict) -> dict:
    """Return ``base`` with ``extra`` merged into it, dict by dict."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def effective_config(overrides: dict | None = None) -> dict:
    """Return the shared settings as the daemon reads them, with ``overrides``.

    Settings are re-read on every call, so a change made in kilix-tts or
    kilix-stt applies to the next turn.
    """
    fresh: dict = {
        "audio": {
            "rate": audio.DEFAULT_RATE,
            "frame_ms": audio.DEFAULT_FRAME_MS,
            "device_in": settings.device_in(),
            "device_out": settings.device_out(),
        },
        "vad": {"silence_ms": settings.stt_silence_ms()},
        "stt": {
            "engine": settings.stt_engine(),
            "model": settings.stt_model(),
            "max_seconds": settings.stt_max_seconds(),
        },
        "tts": {},
        "daemon": {"idle_seconds": IDLE_SECONDS},
    }
    return merge(fresh, overrides or {})
