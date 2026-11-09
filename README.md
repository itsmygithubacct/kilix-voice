# Kilix Voice

Read-aloud and click-to-talk dictation for the terminal: the engine behind
Kilix's speaking-head and microphone tab-bar buttons, plus two standalone
TUIs and the daemon that owns the audio device.

**Status:** ships in Plebian-OS 0.2.3. The release pins it as
`KILIX_VOICE_REF` in `releases/0.2.3.env`, and first boot installs it through
Kilix's `scripts/install-kilix-voice.sh` (also run by `kilix voice install`).
Version 0.1.6.

## What it does

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

## Quick start

On Plebian-OS the commands are already installed and Kilix wraps them:

```sh
kilix speak "Hello from Kilix"     # with no text it reads the active pane
kilix dictate                      # record one utterance, print what it heard
kilix tts                          # read-aloud settings TUI
kilix stt                          # dictation settings, level meter, models
kilix voice status                 # installed runtime and its directories
kilix voice doctor                 # diagnostics, including retained cleanup
```

From a source checkout (Python 3.11 or newer, `espeak-ng` for speech):

```sh
git clone https://github.com/itsmygithubacct/kilix-voice.git
cd kilix-voice
make test          # offline: no microphone, no model, no audio, no network
./kilix-voiced &   # the source tree does not start the daemon for you
./kilix-tts --speak "Hello from Kilix"
./kilix-stt --models
```

`make install PREFIX="$HOME/.local"` installs a self-contained runtime and
`make uninstall PREFIX="$HOME/.local"` removes exactly what it installed.

## Usage

| Command | What it does |
|---|---|
| `kilix-tts --speak TEXT` (`-` reads stdin) | Speak through the daemon; returns once the turn is accepted |
| `kilix-tts --speak - --output FILE.wav` | Render to a new WAV or MP3 file without the daemon |
| `kilix-tts --models`, `--tiers`, `--recommend` | List models, hardware-aware tiers, sizer advice |
| `kilix-tts --interactive --tier neural` | Foreground prompt/play session |
| `kilix-tts --print`, `--set KEY=VALUE` | Read or change read-aloud settings |
| `kilix-stt --models [--json]` | Dictation catalog and installed/default state |
| `kilix-stt --install MODEL --default MODEL` | Install a dictation model (licence-gated) and select it |
| `kilix-stt --setup-default` | First-run setup that Kilix 95 offers |
| `kilix-stt --check-licence MODEL` | Exit 0 if a receipt covers the model, 3 if not |
| `kilix-voiced [--verbose]` | Run the arbiter daemon |

The full command reference, every model family and the dictation engines
(Vosk, VibeVoice, Whisper, Whistle) are in [docs/usage.md](docs/usage.md).

## How it works

### Design in one paragraph

`voicelib/` is **Python standard library only** — there is no third-party
import anywhere, including speech recognition. Recognition binds `libvosk.so`
directly through `ctypes` (seven functions). Speech synthesis uses system
`espeak-ng`, optionally with MBROLA voices, or the separately installed
`kilix-piper-tts` provider. That GPL provider owns its isolated Python runtime
and keeps the pinned Kristin neural model warm; Kilix Voice never imports it.

```
 tab-bar buttons, kilix speak/dictate, kilix-tts, kilix-stt
                         |  private Unix socket (0600, peer-checked)
                         v
                   kilix-voiced  -- one owner of the audio device, half-duplex
            ___________|______________________________
           |                     |                    |
   espeak-ng / MBROLA     provider processes     libvosk.so (ctypes)
                          (Piper, Whisper, ...)
```

The daemon accepts one turn at a time: opening the microphone stops speech,
and a new speech turn replaces the old one. Dictation is click-to-talk, never
presses Enter, refuses a pane whose echo is off, and strips control characters
before injecting text into the PTY. Every action that would fetch model
weights first asks the `kilix-license` authority for a receipt and refuses
with exit 3 when there is none. Settings live in the shared
`~/.local/gpu_terminal/settings.conf`, so the tab-bar buttons, these TUIs and
Kilix 95's Settings app read the same values.

The module contracts are in [DESIGN.md](DESIGN.md); the safety rules and the
licence gate are explained in
[docs/safety-and-licences.md](docs/safety-and-licences.md).

## Documentation

- [DESIGN.md](DESIGN.md) — design principles and module contracts
- [MANAGED-CAPTURE.md](MANAGED-CAPTURE.md) — explicit managed microphone capture
- [docs/usage.md](docs/usage.md) — requirements, run from source, CLI reference, dictation engines, first-run setup
- [docs/safety-and-licences.md](docs/safety-and-licences.md) — safety model, model licences, why a receipt is never shipped
- [docs/settings-and-sizing.md](docs/settings-and-sizing.md) — the shared settings file, data paths, model sizer reports
- [docs/desktop-integration.md](docs/desktop-integration.md) — system voice, Document Reader bridge, owner-scoped speech and dictation
- [docs/cleanup-recovery.md](docs/cleanup-recovery.md) — retained cleanup, process-group handling, accelerator quarantine
- [docs/interactive-tts.md](docs/interactive-tts.md) — foreground TTS auditions and first-use downloads
- [docs/pocket-cpu-audition.md](docs/pocket-cpu-audition.md) — development-only Pocket TTS CPU audition
- [docs/release-notes.md](docs/release-notes.md) — release history

## Related repositories

- [kilix](https://github.com/itsmygithubacct/kilix#readme) — installs this repository at a pinned commit and provides `kilix speak`, `kilix dictate`, `kilix tts`, `kilix stt` and `kilix voice`
- [plebian-os](https://github.com/itsmygithubacct/plebian-os#readme) — pins the release ref and verifies the install at first boot
- [kilix-license](https://github.com/itsmygithubacct/kilix-license#readme) — the licence authority asked before any weight fetch
- [kilix-content](https://github.com/itsmygithubacct/kilix-content#readme) — first-use licence screen and model installs (`kilix models install`)
- [kilix-system-monitor](https://github.com/itsmygithubacct/kilix-system-monitor#readme) — the device-lease component and the model sizer
- [kilix-piper-tts](https://github.com/itsmygithubacct/kilix-piper-tts#readme), [kilix-whisper-stt](https://github.com/itsmygithubacct/kilix-whisper-stt#readme), [kilix-qwen-tts](https://github.com/itsmygithubacct/kilix-qwen-tts#readme) — optional speech providers
- [kilix-bonsai](https://github.com/itsmygithubacct/kilix-bonsai#readme) — installs the shared VibeVoice weights
- [kilix-tui-utils](https://github.com/itsmygithubacct/kilix-tui-utils#readme) — Voice Studio, a front end over these commands
- [kilix-95](https://github.com/itsmygithubacct/kilix-95#readme) — System Voice dialog, Document Reader and first-run dictation setup

## Part of Plebian-OS

Kilix Voice is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
It is pinned in the release manifest and installed for the desktop user at
first boot. The map of every repository and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## Licence

GPL-3.0. See [LICENSE](LICENSE).
