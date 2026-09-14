# kilix-voice — design & module contracts

Read-aloud and dictation for Kilix. Ships two TUI/CLI tools (`kilix-tts`,
`kilix-stt`), one arbiter daemon (`kilix-voiced`), and `voicelib/`.

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
6. **A speak request selects only registered synthesis.** A caller may name a
   catalogued model family, validated voice token, and preset rate. It can
   never supply an executable, model path, URL, or download action.

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
| `voicelib/models.py` | separate canonical STT artifact and request-selectable TTS model catalogs |
| `voicelib/stt.py` | `SttError`, `NullStt`, `VoskStt` (ctypes), `make_stt` |
| `voicelib/tts.py` | `TtsError`, engines, conditioning/chunking, `make_tts`, complete in-memory rendering |
| `voicelib/arbiter.py` | half-duplex policy, single-owner session lock |
| `kilix-voiced` | daemon: control socket, request dispatch, idle exit |
| `kilix-tts` | curses TUI plus arbitrary-text/stdin speech and model-selection CLI |
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

One encoded control request is at most `MAX_REQUEST_BYTES` (192 KiB), which is
below the local `AF_UNIX/SOCK_SEQPACKET` message ceiling. A `speak` request
requires non-empty `text` and may additionally carry `model`, `voice`, and
`rate`. `model` must be in `voicelib.models.TTS_MODEL_IDS`; `voice` must match
`[A-Za-z0-9_+-]{1,32}`; and `rate` must be one of 120, 150, 170, 200, or 240.
Unknown fields are dropped. An accepted speak reply echoes the effective model,
voice, and rate so a new client can detect an older daemon that ignored its
selection and issue a compensating stop.

Replies: `{"ok": true, "id": ...}` or `{"ok": false, "error": "..."}`.
Dictation datagrams: `{"partial": str}`, `{"final": str}`, `{"error": str}`.
Status carries `speech_error` plus a monotonically increasing
`speech_error_serial`, because synthesis and playback finish after the speak
request's connection has closed. A client polls those fields while speaking
and can therefore show each detached worker failure exactly once.

Every accepted job ends in exactly one terminal outcome: `completed`,
`cancelled`, `deadline` or `failed`, with a closed `code` that agrees with it
(none, `cancelled`, `deadline`, or another failure code). The outcome is
recorded once, first writer wins, and is the single source for every channel
that reports it, so no two channels can disagree. A refused request is not a
job; its reply is its only outcome. The job ledger keeps failure prose only,
never spoken or recognised text.

- `status` lists the most recent jobs as `jobs` and names the turn behind the
  latest `speech_error` as `speech_error_turn`. `{"op":"status","job":ID}`,
  with the `turn` a reply carried, adds that job's record as `job`, either
  `{"state":"running"}` or its outcome, and is refused `not-found` for an id
  the daemon does not hold.
- A speak subscriber (`chunk_sock`) that declared `v` of `1.2` or later
  receives one `{"terminal": true, "job", "kind", "outcome", "chunks", ...}`
  message after the last descriptor, then end of stream. A subscriber that
  declared nothing, or an older minor, receives none, so the stream is exactly
  what it was.
- On a chunk stream, `final: true` means only "no further chunk descriptor
  will be published". It is sent while the last clip is still playing and is
  not a success result; the terminal outcome is.
- Every dictation terminal datagram, `final` or `error`, carries the job id as
  `segment`, and a job sends exactly one of them.
- Each chunk descriptor arrives with its audio: one sealed, read-only WAV
  descriptor passed by SCM_RIGHTS, named by `audio_fd` (its index, 0) and
  described by `media_type`, `byte_length` and `sha256`. Its samples are the
  ones the local player was given. A subscriber reading with plain `recv` still
  gets the JSON; the kernel closes the descriptor it did not take. On
  SOCK_STREAM the descriptor rides with the first byte of its line. A subscriber
  that stops reading or goes away loses its subscription, never the audio: the
  daemon never waits on it, and the loss is recorded in the job's outcome as
  `subscriber_lost`.

`stop-dictation` takes an optional `mode`. `finish`, the default, ends
recording and delivers the words heard so far as the job's final. `abort` ends
recording and discards what was heard: final decoding never runs, no final is
sent, and the job's one terminal is an error datagram coded `cancelled`, with
outcome `cancelled`. An abort that arrives after the final was delivered
changes nothing and replies `stopped: false`. The reply echoes the `mode`, so a
client can tell an older daemon, which ignores the field and finishes, from
one that honoured it.

A stop reply for a running turn also says `quiesced`. With `quiesced: true`, no
further audio is fed to the recogniser and no further partial is sent for
that job once the reply is sent. The daemon waits at most one second for a feed
already in progress; `quiesced: false` means one was still running, and its
partial is suppressed. Partials are progress: a receiver that is not reading
loses partials rather than ending the turn, while a receiver that has gone
ends it.

`ingest-audio` hands the daemon audio by descriptor, to be checked and
returned in canonical form. The caller attaches exactly one pre-opened
descriptor -- a pipe's read end, a regular file or a memfd -- by SCM_RIGHTS, and
may declare `sample_format` (`s16le`), `channels` (1), `sample_rate`
(8000-48000 Hz), `container` (`wav` or `raw`), `duration_limit_ms` (up to
600000) and `byte_limit` (up to 32 MiB). Every declaration is optional. The
daemon reads the descriptor within the byte limit, and a pipe within one second
or the request's deadline. It refuses, never truncates, anything that breaks a
declaration or a ceiling: `malformed`, `unsupported`, `too-large` or
`deadline`. A regular file over the byte limit is refused unread, and a device,
socket or directory is refused unread. No path is ever opened on the caller's
behalf, and nothing is resampled or downmixed. The reply carries a canonical
s16le mono WAV as a sealed, read-only descriptor (`audio_fd` 0), with
`sample_rate`, `pcm_bytes`, `byte_length`, `duration_ms`, `sha256` and
`media_type` that match its bytes, and the `job` it was recorded under. It runs
on the control connection: no worker, microphone, arbiter claim, consent gate
or model is involved, so a slow pipe can hold the accept loop for at most one
second.

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
class RenderedSpeech(NamedTuple):
    pcm: bytes; sample_rate: int; chunks: int
    model: str; voice: str; rate: int
class NullTts:   name = "null"; model = "off"
class EspeakTts: name = "espeak"     # espeak-ng --stdout, WAV parsed in memory
    model: str                         # "espeak" or "mbrola"
    def synth(self, text: str) -> tuple[bytes, int]     # (s16le mono pcm, rate)
class PiperTts:  name = "piper"
    model = "piper-en-us-kristin-medium"
    voice = "en_US-kristin-medium"
    def synth(self, text: str) -> tuple[bytes, int]
    def cancel(self) -> None

class SentenceChunker:
    def feed(self, text: str) -> list[str]   # complete sentences
    def flush(self) -> str

def condition_text(text: str, *, max_chars: int | None) -> str
def speech_chunks(text: str, *, max_chars: int | None) -> list[str]
def make_tts(cfg, *, model=None, voice=None, rate=None) -> object
def render_text(text, *, model=None, voice=None, rate=None,
                max_chars=None, cfg=None) -> RenderedSpeech
```

`condition_text` is the read-aloud conditioner and must, in order:
strip ANSI/SGR/OSC/APC sequences; drop kitty graphics payloads entirely;
collapse runs of ≥3 identical box-drawing characters to nothing; collapse
whitespace runs; trim blank leading/trailing lines; truncate at `max_chars`
appending `" …truncated"` when it cuts. It never raises on odd input.

`EspeakTts` uses `mbrola` voices when `KILIX_VOICE_TTS_ENGINE=mbrola`
(`-v mb-<voice>`), and **falls back to plain espeak-ng when the mbrola voice is
unavailable** rather than failing the read. That compatibility fallback applies
to the saved preference only. An explicit per-request `model="mbrola"` is exact
and fails closed rather than claiming it used a model that was unavailable.

`PiperTts` launches only the fixed `kilix-piper-tts` provider executable. The
provider is an independent GPL release closure: it owns Piper, ONNX Runtime,
the checksum-pinned model catalog, and a private persistent worker. Kilix Voice
sends bounded ordinary UTF-8 text, the registered model ID, and a preset rate;
it never sends an executable, path, URL, or raw phoneme block. Cancelling a turn
kills the provider client, whose disconnected socket causes the provider to
kill and recreate an in-flight worker.

### kilix-tts CLI

`--speak TEXT` sends arbitrary text through the same daemon, arbiter, engine,
and sink as read-aloud; `--speak -` reads bounded strict UTF-8 from standard
input. `--model`, `--voice`, and `--rate` are per-request overrides and do not
rewrite shared settings. `--models` lists registered local model families and
their current availability without opening the network or an audio device.
Speech is a standalone action and cannot be combined with settings mutations or
status actions. If acceptance is ambiguous or an older daemon fails to echo an
explicit selection, the client sends a best-effort `stop-speech` before it
reports the error.

`--output FILE` / `--save FILE` changes that same speech action from playback
to synchronous file rendering. Only `.wav` and `.mp3` suffixes are accepted;
MP3 uses an installed local `ffmpeg` or `lame`, never the network. Export does
not contact the daemon or open an audio device. It uses `render_text`, refuses
mixed sample rates, writes mode 0600, and refuses to replace an existing path.
`--install piper-en-us-kristin-medium` is the only TTS network action and is a
standalone explicit command delegated to the fixed provider. `--models` and
status inspection remain download-free.

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
