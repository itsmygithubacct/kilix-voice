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
| `KILIX_VOICE_STT_ENGINE` | `vosk` | `vosk`, `vibevoice`, `whisper`, `off` |
| `KILIX_VOICE_STT_MODEL` | `small-en-us` | `small-en-us`,`lgraph-en-us`,`vibevoice-asr-bitnet`,`whisper-small-en` |
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
7. **No weight fetch without a covering licence receipt** (OD-S, OD-BB;
   OS-V-VERIFY F2). Every route that can cause model weights to be fetched
   calls `licensing.require_covering_receipt()` first. With no covering
   receipt it refuses before the installer is even located — that ordering is
   observable (exit 3 "a human must accept a licence" rather than exit 1 "the
   installer broke" on a machine with no fetcher) and is pinned for all three
   hand-offs by `tests/test_weight_licence.py::LicenceOrderingTests`. No
   process is started, nothing is written under the model store, and the exit
   status is `licensing.LICENCE_REFUSED_EXIT` (3), distinct from an installer
   fault (1). **The gate never writes.** It reads a receipt store that already
   exists and never creates one or changes its mode, on either path; an
   authority that is absent, that raises while importing, or that cannot read
   its own records refuses with the same status rather than escaping as a
   traceback. The Vosk library is Apache-2.0 code, not weights, and is not
   gated. The refusal names kilix-content's first-use flow, by that flow's own
   asset id (`licensing.CONTENT_ASSET_ID`), because it is the one place in the
   stack that shows a licence screen and writes a receipt; a second line names
   `kilix-stt --check-licence MODEL` (or `kilix-tts --check-licence MODEL` for
   the Piper voice), which answers the same question with the same status for
   a fetcher that cannot import this package, and fetches nothing either way.
8. **A receipt is never shipped, vendored, provisioned or built.** The only
   producer of a `kilix.license.receipt/v1` is an acceptance the user made on
   that user's own machine. The schema binds no subject, no timestamp and no
   signature, so any well-formed receipt covers forever for everyone: an image
   carrying one would pass rule 7 on every machine with nobody having seen a
   licence, which is OS-V-VERIFY F2 wearing a compliance badge and contradicts
   OD-S. Shipping a receipt is **not** an acceptable remedy for an image that
   cannot install weights unattended; the acceptable outcomes are that the user
   accepts at first use, or that the weights are not installed. A build-time
   attestation, if one is ever needed, must be a distinct schema (OQ-C4).
   Making a receipt bind the person and the moment belongs to kilix-license,
   the single licence authority (OD-AJ), and is tracked there, not here: this
   gate can only verify what the authority defines.

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
| `voicelib/licensing.py` | the weights gate: `require_covering_receipt`, `LicenseRefused`, `LICENCE_REFUSED_EXIT` |
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
outcome `cancelled`. A stop in either mode that arrives once the job's
terminal is decided -- its final delivered, say, while the recorder and
recogniser are still being torn down -- changes nothing and replies `stopped:
false`. The stop and the final's settlement are decided under one lock. A final
settles only if no abort has won by then, so an abort answered `stopped: true`
always keeps the words from being delivered. The reply echoes the `mode`, so a
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
`deadline`. A regular file over the byte limit is refused unread. A device,
socket or directory is refused unread, and so is a descriptor that grants no
read (write-only, or opened with O_PATH). A read the system refuses is
`unavailable`, or `malformed` when the descriptor itself cannot be read. No path
is ever opened on the caller's behalf, and nothing is resampled or downmixed.
The reply carries a canonical s16le mono WAV as a sealed, read-only descriptor
(`audio_fd` 0), with `sample_rate`, `pcm_bytes`, `byte_length`, `duration_ms`,
`sha256` and `media_type` that match its bytes, and the `job` it was recorded
under.

It answers on the control connection: no worker, microphone, arbiter claim,
consent gate or model is involved. No descriptor can hold the accept loop past
that one second, or the deadline if sooner, plus a quarter-second grace. A pipe
is read through the daemon's own non-blocking description of it, so a sender
that keeps its read end and drains what the daemon was about to read gets a
refusal at the bound rather than a stalled daemon. It also leaves the sender's
own O_NONBLOCK setting alone. Every step that touches the descriptor, fstat
included, runs on a helper thread. A read that has not finished by the bound, a
file on a filesystem the caller serves for example, is refused `unavailable`
(or `deadline`) and left to end on its own. At most two are ever left running;
beyond that a descriptor is refused `busy` without being inspected.

### voicelib/audio.py

`audio.capture_cmd` overrides run in a private process group. Cleanup covers
only descendants that stay in that group. A custom recorder that daemonises
with `setsid()` or `setpgid()` can survive and keep the microphone or a pipe
open; no shipped wrapper does this. These overrides must remain in the group.
SIGKILL of the daemon or meter owner bypasses cleanup. A recorder stalled
without writing can outlive that owner even in its original group, because
it never discovers the closed output pipe. There is no independent
owner-death supervisor. Acoustic silence normally still produces PCM;
output-dependent pipe failure does not cover a stalled recorder.

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
`NullStt`. `vibevoice` gives `VibeVoiceStt`, which always uses the one
VibeVoice model, buffers the turn and runs VibeASR.cpp's `asr_infer --greedy`
once at `end_utterance()` under a bound of 30 s plus 4 s per second of audio.
It never falls back to vosk: a missing runtime or GGUF raises `SttError`
naming what is missing and `kilix stt --install vibevoice-asr-bitnet`.
The runtime is `$KILIX_DATA_HOME/voice/vibeasr/current/bin/asr_infer`, or
`KILIX_VOICE_VIBEASR`; there is no PATH fallback.

`whisper` gives `WhisperStt`, which always uses `whisper-small-en`. Its files
are the kilix-content asset `faster-whisper-small-en`, used where
`kilix models install` publishes it
(`$KILIX_DATA_HOME/desktop-apps/assets/faster-whisper-small-en`), so there is
one copy; a directory at `$KILIX_DATA_HOME/voice/models/whisper-small-en`
wins when present. The engine starts one `kilix-whisper-stt serve` child when it is
built, before the microphone opens, so the model loads while the user speaks;
the child is killed by `close()` and never outlives the recogniser. It is
given the held model descriptors through a private directory of
`/proc/self/fd/N` links (pass_fds). Consent is re-checked once the model has
loaded and after every decode. Each decode sends `{"pcm_bytes": N}` plus the
PCM and reads one JSON line, under a bound of 20 s plus 3 s per second of
audio; the model load is bounded at 60 s. The runtime is
`$KILIX_DATA_HOME/voice/whisper/current/bin/kilix-whisper-stt` (installed by
`kilix voice whisper`), or `KILIX_VOICE_WHISPER`; there is no PATH fallback.
Custom provider wrappers must stay in their original process group.
Whisper and Whistle cleanup covers descendants in that group, including
after the leader exits. A wrapper using `setsid()` or `setpgid()` can escape
and keep a pipe or microphone open. No shipped provider does this; process
groups provide no containment for daemonising wrappers.

Both whole-utterance engines have `provisional()`, and their class sets
`transcribes_at_end`. At a VAD speech end the daemon asks for a provisional
transcript. If it is empty (a start-up pop, a cough), that audio is dropped
and the turn keeps listening. If it holds words, the turn ends, and
`end_utterance()` returns them without decoding again when no audio arrived
since. This is the Vosk `ends_on_words` rule for engines without partial
results.

Consent binds the SHA-256 of every required model file. Those digests are
cached per process only for files whose ctime and mtime were already two
seconds old when hashed (git's racy-index rule), so VibeVoice's 1.7 GB is not
re-read every turn while a rewrite, even of equal length, is always seen.

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

Two sibling sources are on the suite's import path, neither of them copied
here: `LEASE_SRC` (kilix-system-monitor's kilix-device-lease) and
`LICENSE_SRC` (kilix-license). Tests that need them **fail naming the
variable** rather than skipping, because an accelerator engine without a lease,
and a weight fetch without a licence receipt, are exactly what they exist to
rule out. `$(abspath)` resolves both against the checkout, so a linked
worktree passes them explicitly.

`tests/test_weight_licence.py` proves the weights gate against a **spy
installer** — a stand-in `kilix` that records its argv and writes under the
model store as a real fetch would. "Fetches nothing" is observed there, never
read off the source, and two controls keep it honest: a covering receipt must
make the spy run, and a planted copy of each command with the gate line removed
must make every "fetches nothing" assertion fail. The same file pins the parts
of the gate that prose alone used to carry: the receipt's `decision` class in
both directions (with an untampered control), the gate's position ahead of the
fetcher lookup on all three hand-offs (with the planted-gate copy as the
control that the two orderings differ), that the TUI's `i` key is gated as
well as the CLI flag (with a covering-receipt control through the same
keypress), that importing the module creates nothing and imports no authority,
and that every command the refusal names is one these tools accept.

## Registered resource lifecycle

Every daemon-owned resource implements `voicelib.lifecycle.Resource`. The
shared `Lifecycle` implements the states **starting**, **ready**, **in use**,
**closing**, **closed**, and **unproven**. This includes ordinary and managed MicCapture, NullStt,
Vosk, Whisper, Whistle, VibeVoice, eSpeak/MBROLA, Piper, Qwen, the Piper status
owner, NullTts, and the Player with its worker and sinks. `resource_kinds`
registers concrete kinds; `tests/test_lifecycle_matrix.py` compares its
independent matrix with that registry and instantiates both capture factory
variants, so a concrete factory result cannot bypass the inventory.

- A starting operation holds a count before it can allocate. `_publish`
  retains each allocated handle before dropping its own startup count. If
  close won while the allocator ran, publication refuses the handoff and
  signals a late child through its retained group owner. The
  condition lock is never held across an allocator. Close permanently refuses
  new operations; shutdown sets this request even for a constructor already
  doing its own cleanup. Registry insertion and the stopping check share the
  cleanup lock with shutdown's snapshot. Every late registrant is born closed;
  arrivals after the final snapshot log their unproven ownership themselves.
  A starter that resumes after that request gives up
  and cleanup covers all published handles. A tagged lifecycle refusal lets
  synthesis preserve a cancelled turn's outcome while retaining any late
  child for final close; an unrelated launch error remains a failure.
  Capture is a one-shot owner: a
  later turn constructs a new capture.
- A use count covers every native Vosk call, decoder, synthesis exchange,
  capture read, and playback operation. Close can signal a private child to
  interrupt IO, but never frees handles or disposes an active user's state.
  It waits at most `CLOSE_WAIT_S` (0.1 seconds) for counts to drain. At the
  deadline it returns **unproven**, with `cleanup_complete == False`, leaving
  all handles and the registry entry intact. A later retry can prove cleanup.
- Only one disposal is in flight. This guard also wraps constructor failure
  cleanup. The daemon's constructor marker prevents even a second public
  close call during that cleanup; the matrix asserts this separately from
  the resource's disposal guard.
- Ordinary process owners use `processes.GroupPopen`: a private group,
  WNOWAIT exit observation, final signal under the group seal, then reaping.
  On pidfd-capable systems a small isolated Python launcher waits on a private
  inherited socket before executing the requested program. The parent opens
  a non-inherited pidfd, checks its fdinfo PID, checks `/proc/<pid>/status`
  PPid against its own PID and the child's environment against a random
  per-spawn nonce, then checks waitability and pidfd liveness again. Only
  successful verification releases the launcher to exec. The launcher removes
  the nonce at exec by replacing the environment. It reports typed argument,
  protocol and exec failures over the close-on-exec socket; the parent requires
  an exec-attempt marker before accepting EOF. Embedded NUL arguments are
  rejected before spawning. Every falsy `close_fds` value, including positional
  forms, is explicitly unsupported and raises ValueError; callers may select
  inherited descriptors with `pass_fds`.
  A failed exec still reports its original typed exception when cleanup
  cannot finish. The module's failed-launch owner retains the unpublished
  process, pidfd and streams and retries the ordinary group proof, wait and
  disposal on one background worker. An inconclusive scan never seals the
  group or drops ownership. The owner releases the process only after all
  disposal steps succeed; it does not rely on `subprocess._active` to reap it.
  The isolated helper uses built-in modules and a length-prefixed marshal
  request from its trusted parent to avoid importing JSON, socket and signal
  wrappers on every spawn. Short-lived commands cannot exit before their
  identity is established. A launcher killed or reaped during acquisition,
  a wrong descriptor, missing nonce, wrong parent, or unavailable `/proc`
  leaves ownership unproven and never authorizes a signal. Transient
  `pidfd_open` errors fail closed; only a missing API or ENOSYS enables the
  legacy numeric backend.

  This guarantee assumes the trusted launcher/interpreter and local procfs
  are intact, and no untrusted code in the parent can read/forge its nonce or
  alter its descriptors. The default SIGCHLD disposition is required. It does
  not promise availability against a competing reaper: outside reaping is
  logged and retains unproven cleanup. An already-dead launcher is refused;
  an already-dead requested command still has the verified pidfd acquired
  before exec. No trust is inferred just from a same-parent zombie.

  Linux >= 6.9 uses `pidfd_send_signal(fd, SIGKILL, None, 1 << 2)` for the
  original process group. With the group flag unavailable, Linux with
  `pidfd_open` enumerates group members from procfs, opens individual pidfds,
  revalidates each member's group and starttime after opening, then rechecks
  the original leader's still-waitable pidfd after the scan. Only then does
  it signal the collected descriptors, with flags zero. An empty snapshot is
  never completion proof. The termination rule requires a complete fresh scan
  containing the retained leader, in which every group member has a pidfd
  retained from the previous pass that was readable **before** this scan
  began. Each old descriptor must still report that member's PID after the
  scan, so reaping/replacement cannot reuse a previous observation. A zombie
  stat is insufficient: pidfd readability proves all threads have exited.

  Completion additionally requires an unchanged Linux `processes` counter
  from fresh `/proc/stat` reads before and after enumeration and inspection.
  This detects a parent that forks and vanishes before its directory entry
  can even be listed. Linux attaches the new task before incrementing this
  counter and before returning to userspace. Both checks are needed: a
  stable birth count rules out omitted concurrent births; retained readable
  pidfds prove that the listed members cannot fork after the last scan.
  A missing counter or unreadable procfs refuses proof. The counter must be
  the native, unfiltered kernel counter without a full wrap during the bound.
  See [task publication](https://github.com/torvalds/linux/blob/v6.12/kernel/fork.c#L2394-L2438)
  and [counter export](https://github.com/torvalds/linux/blob/v6.12/fs/proc/stat.c#L161-L171).

  A disappearing entry or changed identity makes the scan inconclusive,
  including entries whose group could not be read. A newly observed exited
  member also requires a later scan: it could have forked a live member after
  enumeration. Thus any member able to fork during the final enumeration
  either appears as non-quiescent, or changes the birth counter and makes
  that scan inconclusive. A member omitted because it was born during enumeration has
  such an ancestor. No member known dead before enumeration can create it.
  This relies on complete same-view procfs enumeration and the documented
  group-escape exclusion; hidden processes or a partial procfs mount cannot
  provide this guarantee. It is not a fixed number of empty scans or a delay.

  The one-second bound is unchanged. Failure to obtain this proof raises
  TimeoutError and retains the leader handle, unsealed group, resource and
  lease for retry; neither handle disposal nor lease release is authorized.
  Unrelated process churn can also make scans inconclusive. All temporary
  member descriptors close on success, error and timeout. An
  outside-reaped leader forbids signalling the collected members. Even a
  reap after the final check cannot retarget these descriptors to a new
  numeric group. Linux 5.3's missing `waitid(P_PIDFD)` uses `P_PID` together
  with checks of the retained pidfd's identity. No numeric signal is used
  while a leader pidfd is available.

  Only systems without `pidfd_open` use checked numeric `killpg`. They still
  have a check-to-syscall window and require exclusive child reaping and
  unchanged SIGCHLD throughout spawning, signalling and final wait. That
  legacy path does not provide protection against a competing reaper.
  On all backends, the leader pidfd stays open until final reap, ECHILD never
  becomes successful exit, and ESRCH alone is not cleanup proof. The process
  remains owned until wait succeeds. VibeVoice retains its decoder while
  busy; handing work to it never proves cleanup.
- Explicit `MicCapture(control=CaptureControl(...))` selects the registered
  `managed-capture` kind. Its start, publication, active read and close share
  the same lifecycle counts and one-shot admission. It publishes stderr,
  the CaptureOwner, supervisor and reader before releasing startup. A late
  supervisor, request_stop, and daemon escalation all ask CaptureOwner to
  stop cooperatively. They never kill it as an ordinary recorder group.
  Its dedicated reaper closes the separate recorder session, authenticates
  the cleanup reply and releases or quarantines the microphone lease.
  Capture keeps the owner and all outstanding handles until that proof and
  reader completion; neither a busy start/read nor a false owner stop can
  report successful stop. The default daemon still selects ordinary capture.
- `cleanup_complete` is true only after the shared state reaches closed.
  The daemon checks that proof before marking its owner closed, acknowledging
  an ExecutionGrant, or removing an entry. Acknowledgement failures keep the
  existing retry/quarantine rules. Shutdown starts asynchronous cleanup,
  spends one shared five-second turn-join budget, independently escalates
  still-owned groups, then spends one shared additional one-second cleanup
  budget. It logs every remaining owner as unproven. No close or native free
  runs on the shutdown thread. GroupPopen waits also have a finite default
  ceiling; native calls and kernel calls themselves cannot be forced to return.

The conformance matrix also gates every allocating constructor before its owner
callback, both with and without owned capture still pending at shutdown. It
gates before, inside and after every handle publication, during
active use (both Vosk feed and final-result), during the first blocked close,
and during constructor failure after each publication. It asserts retained
ownership, bounded shutdown and its warning, independent escalation, no busy
handoff proof, one disposal at a time, and final group/reap completion. The
four synthesis variants also cancel inside allocation and assert the terminal
cancelled outcome, with the late child reaped. The
player/sink rows cover worker creation and sink creation separately. Null
engines have no allocating startup or child; the tests state these exceptions.
Constructors that only validate Python configuration cannot leak native
resources; their later allocation/publication paths run the startup matrix.

### Work outside the resource matrix

The matrix does not claim to control group escapes, a kernel-stuck signal
syscall, or native memory safety inside a provider. Process identity tests
exercise outside reaping and SIGCHLD changes, including the forced fallback;
only the backend without pidfd_open retains the numeric window above. These
are ordinary owned-child and mocked signal-routing tests, not a namespace or
tracing qualification. The native Vosk handles are inert fixtures;
other tests exercise a compiled stub library. No physical device, GPU or real
model is required by this suite.

The receiver watcher is a turn-local daemon thread. Its finally path signals
`watch_done` and joins at most `JOIN_TIMEOUT_S`; a stuck watcher can outlive
that join. Shutdown bounds its parent turn's join and closes the control
socket/releases the session, but cannot stop a Python thread. The watcher has
no separate resource entry or escalation handle. The ingest connection
workers are also daemon threads, limited by admission slots and socket IO
bounds; a worker stuck in a filesystem or extension call can outlive shutdown.
Cleanup workers remain attached to their owner state and are reported through
that state. Recorder reader and player worker threads are covered by their
resource's join/reap proof. Python interpreter shutdown ends daemon threads;
it does not constitute a resource/lease acknowledgement.

Vosk's `_load_library` (ctypes/dlopen, symbol binding) precedes the owner
callback. If its native library initializer hangs, no recognizer/model handle
has yet been published to the resource registry. The dictation thread is
joined only to the shared deadline; shutdown can release its session without
proving that initializer complete. Resolution, settings, consent/digest checks
and TTS configuration before registration can likewise block in filesystem or
extension code. They allocate no owned recorder/provider child before the
callback, but are not cancellable native initializers.
Any callback that resumes after shutdown starts is registered with close
already requested under the registry lock. Its constructor cannot allocate
new resources, including after shutdown's last cleanup pass; ownership stays
retained until disposal is proven. Whisper/Whistle and
VibeVoice native runtime/model loading happens in their registered child
processes. External Piper/Qwen service processes belong to their respective
providers; this daemon owns and tests only its client/status processes.

Process API references: [Python subprocess](https://docs.python.org/3/library/subprocess.html)
and [Linux pidfd_send_signal](https://kernel.googlesource.com/pub/scm/docs/man-pages/man-pages/+/master/man/man2/pidfd_send_signal.2).
