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
- **Model weights are fetched only after their licence is accepted.** Every
  install action in this tree asks the `kilix-license` authority for a receipt
  covering that model first. With no covering receipt it refuses, starts no
  installer, writes nothing anywhere, exits **3**, and names the command that
  shows the licence. A machine with no authority installed refuses too — as
  does one whose authority is broken enough to raise — because a machine that
  cannot check a licence must not fetch weights. **A receipt is never
  shipped**: see "A receipt is never shipped" below.

## Model licences

`kilix-license` is the single licence authority. This tree copies none of its
records and mints no receipts of its own: it asks, and refuses when the answer
is no or when there is nobody to ask.

Acceptance happens in kilix-content's first-use flow, which renders the screen,
captures the agreement and writes the receipt before it fetches. kilix-content
files the Vosk weights under their **upstream** ids, so the refusal names those
rather than this catalog's:

```sh
kilix stt --install small-en-us                    # exit 3 without a receipt
kilix models install vosk-model-small-en-us-0.15   # shows the licence, records it
kilix stt --install small-en-us                    # now installs
```

| this catalog | kilix-content asset |
|---|---|
| `small-en-us` | `vosk-model-small-en-us-0.15` |
| `lgraph-en-us` | `vosk-model-en-us-0.22-lgraph` |
| `piper-en-us-kristin-medium` | `piper-en-us-kristin-medium` |
| `qwen3-tts-0.6b-customvoice` | `qwen3-tts-0.6b-customvoice` |
| `vibevoice-asr-bitnet` | `vibevoice-asr-bitnet` |

The two sides are bound by the licence record, not by either id: each asset
names exactly the record digest this authority resolves for the catalog id
beside it.

Receipts are read from the root the licence authority names,
`kilix_license.receipt_store_root()` — `$GPU_TERMINAL_HOME/license-receipts`,
or `$KILIX_LICENSE_RECEIPTS` when that is set — which is where the first-use
flow files them. `$KILIX_VOICE_LICENSE_RECEIPTS` is still read as a legacy
alias, but it moves only this reader, never the writer, and it is ignored
whenever `$KILIX_LICENSE_RECEIPTS` is set. This tree only ever **reads**
that store: it is never created, and its mode is never changed, on the refusing
path or on the covered one.

A fetcher that does not go through this tree cannot import `voicelib`, so it
gets a command instead — one per catalog, so every gated model has a probe that
runs:

```sh
kilix-stt --check-licence small-en-us               # dictation models
kilix-tts --check-licence piper-en-us-kristin-medium # the Piper voice
# exit 0 covered, 3 refused; fetches nothing either way
```

The refusal names both: the acceptance route first, because that is the cure,
and the probe second, because it re-checks without fetching.

### A receipt is never shipped

**The only thing that may produce a receipt is an acceptance the user performed
on that user's own machine.** A receipt must never be vendored into a
repository, baked into an image, provisioned onto a machine, or written by a
build.

This has to be said out loud because the gate cannot tell the difference. A
`kilix.license.receipt/v1` binds the licence record, the licence text digest,
the licensor, the binding conditions and the decision class — it carries **no
subject, no timestamp and no signature**. Any well-formed receipt in the store
therefore covers forever, for every user, on every machine. An image that
shipped one would pass this gate on every unit sold while nobody had ever seen
a licence, and the audit surface would read "gated": the hole OS-V-VERIFY F2
reported, restored in a form that looks compliant. OD-S is explicit that the
user "gives an explicit acceptance before download".

So shipping a receipt is **not** an acceptable remedy for an image that cannot
install weights unattended. The acceptable outcomes are exactly two: the user
accepts at first use, or the weights are not installed. If a build ever needs a
build-time attestation of its own, OQ-C4 already rules that it must be a
distinct schema and never a `kilix.license.receipt/v1`.

Making a receipt bind the person and the moment is the deeper fix, and it is
**not this repository's to make**: kilix-license owns the receipt schema as the
single licence authority (OD-AJ), and that work is tracked there. This gate
verifies what the authority defines and cannot bind more than the authority
binds.

**The library is not weights.** `libvosk.so` is Apache-2.0 code. Loading it,
probing for it, reporting on it and installing it need no receipt, and none of
those paths touch the gate: `--print`, `--models`, `--models --json` and the
whole read-aloud side keep working with no receipt and no authority. Only the
three actions that cause weights to be fetched are gated — `kilix-stt
--install` for the Vosk models and for the shared VibeVoice weights, and
`kilix-tts --install` for the Piper voice.

One coupling is outside this repository: at the pinned ref,
`kilix/scripts/install-kilix-voice.sh` fetches the Vosk wheel and the model in
one block, and its `--without-dictation` flag skips both. So refusing the
weights on that leg also withholds the library. Keeping the library installable
on its own needs a library-only leg in that installer; nothing here can do it.

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
`--model mbrola`, `--model piper-en-us-kristin-medium`, and
`--model qwen3-tts-0.6b-customvoice` are the registered local model families. A request
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

For foreground prompt/play/replay sessions, offline resident Qwen auditions,
and first-use Qwen downloads, see [Interactive TTS](docs/interactive-tts.md).
`kilix-tts --tiers` lists hardware-aware choices. In an interactive terminal,
`kilix-tts --interactive --tier neural` starts Piper Kristin and installs its
weights on first selection after showing the licence and waiting for a keypress.
On a fresh Kilix installation, `kilix tts --interactive --tier neural` also
installs the pinned Piper runtime on demand. Neither listing nor selecting a
different tier starts a Piper download.
`kilix tts --interactive --tier qwen-cpu` lazily installs a locked CPU runtime
on x86_64 Debian, then offers the catalogued 0.6B model after its first-use
licence notice. On eligible hardware, `kilix tts --interactive --tier qwen-gpu`
also lazily installs a locked CUDA/FlashAttention runtime; neither tier is an
automatic install or fallback.
`kilix tts --interactive --tier qwen-base-gpu` shares that GPU runtime but
uses separately catalogued 0.6B Base weights and a fixed synthetic eSpeak
reference; it never clones a person's voice.
`python3 bench/tts_reply.py --output new-result.json` measures cold and warm
Piper prompt-to-playable-audio reply time through the Kilix provider path.
The new first-use tests also require a 0.2.2 `kilix-content` source tree; pass
`CONTENT_SRC=/path/to/kilix-content/src` alongside `LICENSE_SRC` when testing
separate release worktrees.

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
