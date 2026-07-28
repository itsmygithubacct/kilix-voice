"""Cross-module event vocabulary. Frozen contract — see DESIGN.md.

Constants rather than an enum: these values travel through JSON on the control
socket and into log lines, so their wire form is the value itself.
"""

from __future__ import annotations

from dataclasses import dataclass

# Vad.feed() returns exactly one of these per audio frame.
VAD_SILENCE = "silence"
VAD_MAYBE = "maybe"                  # above the floor, not yet long enough
VAD_SPEECH_START = "speech_start"
VAD_SPEECH = "speech"
VAD_SPEECH_END = "speech_end"

VAD_EVENTS = (VAD_SILENCE, VAD_MAYBE, VAD_SPEECH_START, VAD_SPEECH,
              VAD_SPEECH_END)

# Turn.kind: what the machine did, not who spoke.
TURN_SPEECH = "speech"        # text read aloud to the user
TURN_DICTATION = "dictation"  # speech recognised from the user


@dataclass
class Turn:
    """One completed voice interaction, for history and for the TUIs' log."""

    kind: str            # TURN_SPEECH | TURN_DICTATION
    text: str
    t: float             # unix time the turn completed
    engine: str = ""     # engine that produced it, e.g. "espeak", "vosk"
    partial: bool = False  # cut short by barge-in, stop, or the turn timeout
