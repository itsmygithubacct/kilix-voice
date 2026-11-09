# Requirements and command reference

Reference for [Kilix Voice](../README.md).

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

Through the Kilix launcher, these foreground sessions lazily install their
runtime and model, present the model's first-use terms, and reuse verified
weights on subsequent launches:

```bash
kilix tts --interactive --tier neural
kilix tts --interactive --download-pocket
kilix tts --interactive --download-qwen qwen3-tts-0.6b-customvoice
kilix tts --interactive --download-qwen qwen3-tts-0.6b-base --synthetic-reference
kilix tts --interactive --download-qwen qwen3-tts-1.7b-voicedesign --description 'A warm clear voice'
```

The Base session uses a generated eSpeak reference. VoiceDesign accepts
`/description TEXT` to change the voice. All sessions support `/save FILE.wav`.
Direct Qwen sessions default to CPU; `--device cuda:0` explicitly chooses the
optional GPU runtime. These foreground sessions do not change daemon defaults.
The native Pocket Q8 research variant is excluded.

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
`--model mbrola`, `--model piper-en-us-kristin-medium`, and
`--model qwen3-tts-0.6b-customvoice` are the registered local model families. A request
may also override the voice and one of the shared WPM presets without changing
the saved read-aloud settings. Model IDs are catalogued: callers cannot supply
an executable, model path, URL, or download action. An explicit MBROLA request
fails if its voice is unavailable; the longstanding saved MBROLA preference
keeps its eSpeak fallback. Use `--speak -` for strict UTF-8 standard input.
Piper supports `en_US-kristin-medium` and `kilix-female-02` through CPU-only
inference. The 0.2.2 default is `kilix-female-02` checkpoint 80. With engine
`piper`, an unset voice or a saved generic eSpeak name selects that voice for
Avatar and other callers that omit a voice/model override. An explicit model
remains exact, including Kristin.
Female-02 requires the named-voice provider and its checksum-pinned local export;
use `kilix-piper-tts install kilix-female-02 --model-file FILE --config-file FILE`.
Unavailable named voices fail clearly without speaking through another model.
`--install piper-en-us-kristin-medium` is a separate explicit
network action delegated to the checksum-pinned `kilix-piper-tts` catalog;
listing models, opening the TUI, and speaking with other engines never
download it.
Qwen is an explicit candidate only: `--model qwen3-tts-0.6b-customvoice`
requires the `kilix-qwen-tts` client executable from Kilix's managed
`voice/qwen-client/current` generation, the daemon's `PATH`, or
`KILIX_QWEN_TTS`, and a running local provider with that model
installed through `kilix-content`
and its licence receipt. It does not fetch weights or start a provider.
Its default named voice is `Vivian`; `--voice` selects a provider voice ID,
and `--rate` is unsupported. It is not a saved default or a release-qualified
runtime yet.
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
Kilix Bonsai. `kilix stt --install vibevoice-asr-bitnet` pulls those weights and
then builds the pinned VibeASR.cpp runtime (`kilix voice vibeasr`, needs a C/C++
toolchain and cmake). VibeVoice is multilingual and more accurate on natural
speech than Vosk, but it transcribes the whole turn when it ends: there are no
live partials, and a turn takes roughly its own length again to transcribe on
four threads. Decoding is greedy, so the same audio gives the same text. The
turn's audio is written to a private temporary WAV under the voice session
directory and removed as soon as transcription finishes.

`whisper-small-en` is Whisper small English through the pinned
`kilix-whisper-stt` provider (faster-whisper, int8 on the CPU). On the
reference dictation it made about a third of the Vosk small model's errors
and answered in about a second per sentence. It is the model sizer's default
on a machine with AVX2, 8 logical CPUs and 16 GiB of RAM.

`whistle` selects Cactus Whistle through a pinned local Needle 3 native
library on Linux x86-64 (glibc). `kilix stt --install whistle` shows the
weight licence through Kilix Content and installs the native runtime;
`kilix stt --default whistle` selects it explicitly. It is an optional
engine; installing it does not replace the saved default or the model
sizer's Whisper recommendation.

Whistle dictation is English, without keyword bias. Turns up to 30 seconds
use whole-clip decoding; longer turns use the native streaming decoder
and can take substantially longer. The provider returns text at the end of
the turn, keeps model bytes bound to capture consent, and performs no
network requests. The native runtime chooses its own thread count.
The normal dictation deadline still applies: long turns can time out on
slower CPUs. Prefer Whisper for long dictation.

```sh
kilix stt --install whisper-small-en
```

The command above runs kilix-content's `kilix models install
faster-whisper-small-en`, which shows the licence and takes the typed agreement
before anything is fetched. It then installs the provider (`kilix voice
whisper`). Weights that are already installed are not fetched again. Like
VibeVoice, Whisper transcribes the turn when it ends. Its provider is started
at the beginning of each turn, so the model loads while you speak.

`kilix stt --setup-default [--result FILE]` is the first-run setup that Kilix 95
offers. It:

1. asks the model sizer which model suits this computer, falling back to
   `small-en-us` when the sizer cannot answer;
2. shows the plan and asks to go ahead;
3. runs the licence screen and the install;
4. asks permission for dictation to use the microphone, and records the
   consent;
5. makes the model the dictation default;
6. names the dictation hotkey (Ctrl+Shift+D).

With `--result FILE` it writes `{"status": "ready"|"declined"|"failed", "model":
ID, "detail": TEXT}` atomically when it ends. It exits 0, 1 or 2 to match that
status.

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
