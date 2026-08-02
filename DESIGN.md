# kilix-voice — design & module contracts

Read-aloud and dictation for Kilix. Ships two TUIs (`kilix-tts`, `kilix-stt`),
one arbiter daemon (`kilix-voiced`), and `voicelib/`.

This file is the **authoritative contract**. Implement what is written here.
If a contract seems wrong, implement it as written and flag the concern —
do not edit shared or foreign files.

## Principles

- **Python 3.11+, stdlib only.** There is **no third-party import anywhere**,
  including STT: `stt.py` binds `libvosk.so` through `ctypes`. No `pip`, no
  venv, no wheel. Runs under system Python.
- **Engines are synchronous primitives; the daemon owns all threads.** Engine
  classes never spawn threads except where specified (`MicCapture` reader,
  `Player` worker). This keeps every engine testable inline.
- **Everything is testable offline**: no network, no microphone, no model, no
  audio device, and **no audible output** in tests. External processes are
  fakes injected via config command overrides.
- **A missing optional dependency degrades a feature, never blocks a launch.**
  Every failure path returns an actionable message: what failed + what to do.

## Audio format (frozen)

s16le (signed 16-bit little-endian) **mono** PCM everywhere. Capture and STT
run at `audio.rate` = **16000 Hz**, frame `audio.frame_ms` = **20 ms**
(⇒ 640 bytes/frame). TTS engines return PCM at their own native rate; the
Player is told the rate per clip.

## Paths

All resolved by `voicelib/paths.py`, honouring the Kilix environment:

| Purpose | Path |
|---|---|
| shared settings | `$GPU_TERMINAL_SETTINGS_FILE` or `$GPU_TERMINAL_HOME/settings.conf` (default `~/.local/gpu_terminal/settings.conf`) |
| session (sockets) | `$KILIX_SESSION_HOME/voice/` (default `~/.local/gpu_terminal/kilix/session/voice`), mode 0700 |
| data (models, lib) | `$KILIX_DATA_HOME/voice/` (default `~/.local/gpu_terminal/kilix/data/voice`), mode 0700 |
| control socket | `<session>/control.sock`, mode 0600 |
| dictation socket | `<session>/dictate-<paneid>.sock` — **created by the kitty fork**, never by us |
| libvosk | `<data>/lib/current/libvosk.so` |
| models | `<data>/models/<catalog-id>/` |

Never create a socket outside `<session>`. Never accept a `sock` path that does
not resolve inside `<session>`.

## Settings vocabulary

`voicelib/settings.py` reads the **shared** Kilix settings file — the same
`KEY=value` file Kilix's SDK writes. It is the single source of truth; this
repo never invents a private config file.

| Key | Default | Choices |
|---|---|---|
| `KILIX_CHROME_SPEAK` | `1` | bool |
| `KILIX_CHROME_DICTATE` | `1` | bool |
| `KILIX_VOICE_TTS_ENGINE` | `espeak` | `espeak`, `mbrola`, `off` |
| `KILIX_VOICE_TTS_VOICE` | `en-us` | token `[A-Za-z0-9_+-]{1,32}` |
| `KILIX_VOICE_TTS_RATE` | `170` | `120`,`150`,`170`,`200`,`240` |
| `KILIX_VOICE_TTS_EXTENT` | `screen` | `screen`,`scrollback`,`selection` |
| `KILIX_VOICE_TTS_MAX_CHARS` | `4000` | `1000`,`4000`,`16000`,`unlimited` |
| `KILIX_VOICE_STT_ENGINE` | `vosk` | `vosk`, `vibevoice`, `off` |
| `KILIX_VOICE_STT_MODEL` | `small-en-us` | `small-en-us`,`lgraph-en-us`,`vibevoice-asr-bitnet` |
| `KILIX_VOICE_STT_SUBMIT` | `never` | `never`, `confirm` — **no `always` value exists** |
| `KILIX_VOICE_STT_MAX_SECONDS` | `30` | `15`,`30`,`60`,`120` |
| `KILIX_VOICE_STT_SILENCE_MS` | `900` | `500`,`900`,`1500` |
| `KILIX_VOICE_STT_PUNCTUATION` | `1` | bool |
| `KILIX_VOICE_DEVICE_IN` | `default` | `default` or PulseAudio source token |
| `KILIX_VOICE_DEVICE_OUT` | `default` | `default` or sink token |
| `KILIX_VOICE_HISTORY` | `off` | `off`, `on` |

Unrecognised values fall back to the default — never coerce, never guess.
Booleans are false for `"" 0 no false off disabled` (case-insensitive).

## Safety rules (load-bearing, not preferences)

1. **The microphone is click-to-talk only.** Capture opens on an explicit
   request and closes on stop/timeout/silence. **No pre-roll buffer.**
2. **Dictation never submits.** `stt_submit()` is `never` or `confirm`; there
   is no `always`. Final text is delivered **without** a trailing newline, and
   any trailing newline in the recognised text is stripped.
3. **Sanitise before injection.** Strip all control characters below 0x20
   except space; strip DEL (0x7f); strip CSI/OSC introducers. A recogniser
   should never emit these, but the sink is a PTY.
4. **Nothing leaves the machine.** No network calls at runtime, ever.
5. Sockets 0600, directories 0700, `SO_PEERCRED` uid checked on every accept.

## Files & ownership

| File | Contents |
|---|---|
| `voicelib/util.py` | `cfg_get`, `rms16`, `parse_wav_bytes`, `write_wav`, `which`, `repo_root` |
| `voicelib/paths.py` | path resolution + `ensure_private_dir` |
| `voicelib/settings.py` | shared-settings reader, typed accessors, choice validation |
| `voicelib/events.py` | VAD event constants, `Turn` dataclass |
| `voicelib/protocol.py` | control/dictation message encode/decode + validation |
| `voicelib/audio.py` | `build_capture_cmd`, `MicCapture`, `build_play_cmd`, `Player`, `AudioError` |
| `voicelib/vad.py` | `Vad` |
| `voicelib/stt.py` | `SttError`, `NullStt`, `VoskStt` (ctypes), `make_stt` |
| `voicelib/tts.py` | `TtsError`, `NullTts`, `EspeakTts`, `SentenceChunker`, `condition_text`, `make_tts` |
| `voicelib/arbiter.py` | half-duplex policy, single-owner session lock |
| `kilix-voiced` | daemon: control socket, request dispatch, idle exit |
| `kilix-tts` | curses TUI |
| `kilix-stt` | curses TUI |

## Contracts

### voicelib/util.py

```python
def cfg_get(cfg: dict, path: str, default=None)   # "a.b.c" dotted lookup
def rms16(frame: bytes) -> float                  # RMS of s16le, 0.0 on empty
def parse_wav_bytes(data: bytes) -> tuple[bytes, int]   # (pcm s16le mono, rate); raises ValueError
def write_wav(pcm: bytes, rate: int) -> bytes
def which(name: str) -> str | None
```

### voicelib/settings.py

```python
SPEC: dict[str, tuple[str, tuple[str, ...] | None]]   # key -> (default, choices|None)
def load(path: str | None = None) -> dict[str, str]
def value(key: str, path=None) -> str        # validated against SPEC choices
def enabled(key: str, path=None) -> bool
def tts_engine(path=None) -> str
def tts_rate(path=None) -> int
def tts_max_chars(path=None) -> int | None   # None == unlimited
def stt_engine(path=None) -> str
def stt_submit(path=None) -> str
def stt_max_seconds(path=None) -> int
def stt_silence_ms(path=None) -> int
def update(changes: dict[str, object], path=None) -> str   # atomic rewrite, 0600
```

`update()` must preserve unknown keys and comments already in the file — it is
a shared file owned by Kilix, not by us.

### voicelib/protocol.py

Line-delimited JSON. One message per `send`.

```python
class ProtocolError(ValueError): ...
def encode(msg: dict) -> bytes                  # utf-8, single trailing \n
def decode(raw: bytes) -> dict                  # raises ProtocolError
def validate_request(msg: dict, session_dir: str) -> dict
    # requires 'op' in {speak, stop-speech, dictate, stop-dictation, status}
    # 'sock' (dictate only) MUST resolve inside session_dir -> else ProtocolError
    # returns a normalised copy with defaults applied
```

Replies: `{"ok": true, "id": ...}` or `{"ok": false, "error": "..."}`.
Dictation datagrams: `{"partial": str}`, `{"final": str}`, `{"error": str}`.
Status carries `speech_error` plus a monotonically increasing
`speech_error_serial`, because synthesis and playback finish after the speak
request's connection has closed. A client polls those fields while speaking
and can therefore show each detached worker failure exactly once.

### voicelib/audio.py

```python
def build_capture_cmd(cfg) -> list[str]
    # cfg audio.capture_cmd (list, "{rate}"/"{device}" substituted) if set;
    # else parec: ["parec","--format=s16le","--rate={rate}","--channels=1","--latency-msec=30"]
    #             + ["-d", device] when device != "default"
    # else arecord: ["arecord","-q","-f","S16_LE","-r","{rate}","-c","1","-t","raw"]
    # else raise AudioError with install guidance.

class MicCapture:
    def start(self) -> None            # subprocess + reader thread
    def read(self, timeout=None) -> bytes | None   # exactly one frame, or None
    def stop(self) -> None
    @property
    def rate(self) -> int
    @property
    def frame_bytes(self) -> int       # rate * frame_ms/1000 * 2

def build_play_cmd(cfg, rate: int) -> list[str]    # pacat/aplay, device-aware
class Player:
    def play(self, pcm: bytes, rate: int) -> None  # queue, non-blocking
    def stop(self) -> None                          # cancel within ~100 ms
    def wait(self, timeout=None) -> bool
    @property
    def playing(self) -> bool
```

### voicelib/vad.py

```python
class Vad:
    def feed(self, frame: bytes) -> str    # one of events.VAD_*
    def reset(self) -> None
    def set_suppressed(self, on: bool) -> None   # echo gate while speaking
    @property
    def level(self) -> float
    @property
    def noise_floor(self) -> float
```

Speech opens after ~200 ms sustained above the adaptive floor; closes after
`silence_ms`. While suppressed, require ~350 ms sustained.

### voicelib/stt.py — ctypes, no wheel

```python
class SttError(RuntimeError): ...

class NullStt:
    name = "null"; supports_partials = True
    def start_utterance(self); def feed(self, frame) -> str | None
    def end_utterance(self) -> str; def close(self)

class VoskStt:
    name = "vosk"; supports_partials = True
```

`VoskStt` loads `libvosk.so` with `ctypes.CDLL` and declares **exactly** these
seven prototypes with explicit `argtypes`/`restype`:

```
vosk_set_log_level(int) -> None
vosk_model_new(c_char_p) -> c_void_p
vosk_model_free(c_void_p) -> None
vosk_recognizer_new(c_void_p, c_float) -> c_void_p
vosk_recognizer_free(c_void_p) -> None
vosk_recognizer_accept_waveform(c_void_p, c_char_p, c_int) -> c_int
vosk_recognizer_partial_result(c_void_p) -> c_char_p
vosk_recognizer_final_result(c_void_p) -> c_char_p
```

Results are **borrowed** `const char *` — copy (`bytes(...)`/`.decode()`)
before the next call into the library. A missing library or model raises
`SttError` with an actionable message, never a bare `OSError`.

`make_stt(cfg, rate)` dispatches on `KILIX_VOICE_STT_ENGINE`; `off`/unknown →
`NullStt`. `vibevoice` is **not implemented in this phase** — `make_stt` must
raise `SttError` naming it as a later phase, not silently fall back.

### voicelib/tts.py

```python
class TtsError(RuntimeError): ...
class NullTts:   name = "null"
class EspeakTts: name = "espeak"     # espeak-ng --stdout, WAV parsed in memory
    def synth(self, text: str) -> tuple[bytes, int]     # (s16le mono pcm, rate)

class SentenceChunker:
    def feed(self, text: str) -> list[str]   # complete sentences
    def flush(self) -> str

def condition_text(text: str, *, max_chars: int | None) -> str
def make_tts(cfg) -> object
```

`condition_text` is the read-aloud conditioner and must, in order:
strip ANSI/SGR/OSC/APC sequences; drop kitty graphics payloads entirely;
collapse runs of ≥3 identical box-drawing characters to nothing; collapse
whitespace runs; trim blank leading/trailing lines; truncate at `max_chars`
appending `" …truncated"` when it cuts. It never raises on odd input.

`EspeakTts` uses `mbrola` voices when `KILIX_VOICE_TTS_ENGINE=mbrola`
(`-v mb-<voice>`), and **falls back to plain espeak-ng when the mbrola voice is
unavailable** rather than failing the read.

### voicelib/arbiter.py

```python
class Arbiter:
    def acquire_session(self) -> None    # O_EXCL lock in session dir + liveness
    def release(self) -> None
    def begin_speech(self, id) / end_speech(self, id)
    def begin_listen(self, id) / end_listen(self, id)
    @property
    def speaking(self) -> bool
    @property
    def listening(self) -> bool
```

Half-duplex policy: opening the mic **cancels any in-flight speech first**
(barge-in). Starting speech while listening is refused with a clear error.

## Testing

`make test` runs `python3 -m unittest discover -s tests -v`. Tests must pass
with no `libvosk.so`, no model, no `espeak-ng`, no audio server and no network.
Build a **stub `.so`** in a fixture (compile a few lines of C with `cc` at test
time, skip the test if no compiler) to exercise the ctypes binding.
