# Kilix Voice

Kilix Voice gives a terminal two things it normally lacks: it can **read what a
pane is showing out loud**, and it can **type what you say into a pane**. It
ships the engine behind Kilix's speaking-head and microphone tab-bar buttons,
plus two standalone TUIs and the small daemon that owns the audio device.

Everything runs locally. No audio, text, or transcript ever leaves the machine.

- `kilix-tts` — read-aloud settings plus a scriptable speech command for
  arbitrary text, standard input, per-request model/voice/rate selection, and
  WAV/MP3 file export
- `kilix-stt` — dictation: input device, model, a live level meter and voice
  activity readout for working out why it cannot hear you, plus explicit model
  installation/default selection
- `kilix-voiced` — the arbiter: one owner of the audio device, half-duplex, so
  opening the microphone stops speech instead of transcribing it

## Design in one paragraph

`voicelib/` is **Python standard library only** — there is no third-party
import anywhere, including speech recognition. Recognition binds `libvosk.so`
directly through `ctypes` (seven functions). Speech synthesis uses system
`espeak-ng`, optionally with MBROLA voices, or the separately installed
`kilix-piper-tts` provider. That GPL provider owns its isolated Python runtime
and keeps the pinned Kristin neural model warm; Kilix Voice never imports it.

## Safety model

These are not preferences; they are enforced by the code and covered by tests.

- **The microphone is click-to-talk.** Capture opens on an explicit action and
  closes on stop, timeout, or silence. Nothing is buffered before you ask, so
  there is no rolling recording.
- **Dictation never presses Enter.** The submit policy is `never` or `confirm`;
  there is deliberately no `always`. Recognised text is inserted without a
  trailing newline for you to read before you run it.
- **Dictation refuses a hidden prompt.** If the target pane has echo disabled —
  a password prompt — the request is refused rather than typed.
- **Text is sanitised before injection.** Control characters and escape
  sequence introducers are stripped; the destination is a PTY.
- Sockets are private to your user (mode 0600 inside a 0700 directory) and the
  daemon checks peer credentials on every connection.

## Requirements

- Python 3.11 or newer
- `espeak-ng` for read-aloud (`mbrola` plus a voice such as `mbrola-us1` is an
  optional quality tier)
- `kilix-piper-tts` plus its explicitly installed
  `piper-en-us-kristin-medium` model for the optional local neural tier
- `ffmpeg` or `lame` only when exporting MP3; WAV export needs no encoder
- PulseAudio or PipeWire tools — `parec`/`pacat`, or ALSA's `arecord`/`aplay`
- `libvosk.so` and a model for dictation, built and fetched by Kilix's pinned
  installer

Read-aloud works without any recognition support, and dictation works without
any synthesis support. A missing piece disables that one feature and says so;
it never blocks startup.

## Run from source

```bash
git clone https://github.com/itsmygithubacct/kilix-voice.git
cd kilix-voice
make test        # offline: no microphone, no model, no audio, no network
./kilix-tts      # read-aloud settings
./kilix-stt      # dictation settings and level meter
```

Both TUIs also work as plain CLIs:

```bash
./kilix-tts --print
./kilix-tts --set wpm=200
./kilix-tts --models
./kilix-tts --install piper-en-us-kristin-medium
./kilix-tts --speak "Hello from Kilix"
./kilix-tts --speak "Hello from Kristin" \
  --model piper-en-us-kristin-medium
printf '%s\n' "Text from an agent" | \
  ./kilix-tts --speak - --model mbrola --voice us1 --rate 200
./kilix-tts --speak - --model espeak --voice en-us \
  --output kilix-test.wav < examples/tts-export-test.txt
./kilix-tts --speak - --model espeak --save kilix-test.mp3 \
  < examples/tts-export-test.txt
./kilix-stt --models
./kilix-stt --models --json
./kilix-stt --install lgraph-en-us --default lgraph-en-us
./kilix-stt --set stt_submit=confirm
```

`kilix-tts --speak` returns as soon as `kilix-voiced` accepts the turn; speech
continues in the background and a new turn replaces it. `--model espeak`,
`--model mbrola`, and `--model piper-en-us-kristin-medium` are the registered
local model families. A request
may also override the voice and one of the shared WPM presets without changing
the saved read-aloud settings. Model IDs are catalogued: callers cannot supply
an executable, model path, URL, or download action. An explicit MBROLA request
fails if its voice is unavailable; the longstanding saved MBROLA preference
keeps its eSpeak fallback. Use `--speak -` for strict UTF-8 standard input.
The Piper model has the fixed `en_US-kristin-medium` voice and CPU-only
inference. `--install piper-en-us-kristin-medium` is a separate explicit
network action delegated to the checksum-pinned `kilix-piper-tts` catalog;
listing models, opening the TUI, and speaking with other engines never
download it.
When running from a source checkout, start `./kilix-voiced` first; the main
`kilix speak` wrapper starts an installed daemon on demand.

`--output FILE` (also spelled `--save FILE`) renders the same conditioned text
directly to a new `.wav` or `.mp3` file instead of playing it. Export does not
need the daemon and never opens an audio device. The suffix selects the format;
MP3 encoding stays local through `ffmpeg` or `lame`. Output is mode 0600 and an
existing file is never overwritten implicitly. The self-describing paragraph
in `examples/tts-export-test.txt` exercises arbitrary UTF-8 stdin, selection,
synthesis, and both containers.

Opening `kilix-stt` only lists local state; it never downloads a model. On the
Models tab, `i` hands the terminal to Kilix's checksum-pinned lazy installer,
Enter pairs the selected model with its matching recognizer, and `s` saves the
default. The catalog is `small-en-us` and `lgraph-en-us` for the runnable Vosk
engine, plus `vibevoice-asr-bitnet`, whose shared weights are installed through
Kilix Bonsai. VibeVoice can be installed and selected for forward compatibility,
but this voice runtime does not yet run it and says so in both TUI and CLI output.

`voicelib.models` is the canonical in-process catalog. Cross-process consumers
use `kilix-stt --models --json`, whose `kilix.speech.models/v1` document
reports immutable catalog metadata, local installed/default state, and the
common install-and-default argv without opening the network. Installation
remains a separate explicit action. This versioned CLI document is the 0.1.9
compatibility boundary for the terminal chrome and desktop surfaces; consumers
must reject an unknown schema rather than guessing at fields.

`make install PREFIX=/path` creates a self-contained runtime: the three
commands land in `bin/`, while their exact `voicelib` package and `VERSION`
land in `lib/kilix-voice/`. Installed commands therefore do not depend on the
source checkout or an ambient `PYTHONPATH`. `make uninstall PREFIX=/path`
removes that exact runtime and refuses to remove files changed since install.

## Release history

- **0.1.6** — add the isolated persistent Piper provider contract, explicit
  pinned Kristin installation, cancellable neural speech, and WAV/MP3 export
  through the existing arbitrary-text CLI.
- **0.1.5** — render arbitrary speech to private, no-overwrite WAV or MP3
  files, with the daemon's conditioning and model-selection semantics.
- **0.1.4** — add arbitrary-text/stdin speech, safe per-request TTS
  model/voice/rate selection, model discovery, and acknowledgement-loss stop
  compensation to `kilix-tts`.
- **0.1.3** — centralize the speech-model catalog and publish its download-free
  `kilix.speech.models/v1` JSON control-plane contract.
- **0.1.2** — expose asynchronous synthesis/playback failures through daemon
  status so detached read-aloud errors remain visible in Kilix.
- **0.1.1** — make installed runtimes self-contained by packaging `voicelib`
  and add an install-path execution regression test.
- **0.1.0** — initial local read-aloud and click-to-talk dictation engine.

## Settings

Kilix Voice does not have its own configuration file. It reads and writes the
shared GPU Terminal settings file — the same one Kilix, Kilix 95, and Pleb use
— so the tab-bar buttons, these TUIs, `kilix settings`, and Kilix 95's Settings
app can never disagree with each other:

```
~/.local/gpu_terminal/settings.conf
```

Models and the recognition library live under `~/.local/gpu_terminal/kilix/data/voice/`,
and runtime sockets under `~/.local/gpu_terminal/kilix/session/voice/`. Both are
private to your user.

## Licence

GPL-3.0. See [LICENSE](LICENSE).
