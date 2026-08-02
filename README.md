# Kilix Voice

Kilix Voice gives a terminal two things it normally lacks: it can **read what a
pane is showing out loud**, and it can **type what you say into a pane**. It
ships the engine behind Kilix's speaking-head and microphone tab-bar buttons,
plus two standalone TUIs and the small daemon that owns the audio device.

Everything runs locally. No audio, text, or transcript ever leaves the machine.

- `kilix-tts` — read-aloud: engine, voice, speaking rate, how much of the pane
  to read, and a test phrase
- `kilix-stt` — dictation: input device, model, a live level meter and voice
  activity readout for working out why it cannot hear you
- `kilix-voiced` — the arbiter: one owner of the audio device, half-duplex, so
  opening the microphone stops speech instead of transcribing it

## Design in one paragraph

`voicelib/` is **Python standard library only** — there is no third-party
import anywhere, including speech recognition. Recognition binds `libvosk.so`
directly through `ctypes` (seven functions), so there is no wheel, no virtual
environment, and nothing to `pip install`. Speech synthesis shells out to
`espeak-ng`, optionally through `mbrola` voices. Both are ordinary packages
from your distribution.

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
./kilix-stt --set stt_submit=confirm
```

`make install PREFIX=/path` creates a self-contained runtime: the three
commands land in `bin/`, while their exact `voicelib` package and `VERSION`
land in `lib/kilix-voice/`. Installed commands therefore do not depend on the
source checkout or an ambient `PYTHONPATH`.

## Release history

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
