# Settings and model sizing

Reference for [Kilix Voice](../README.md).

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

## Model sizing

Both tools ask the shared `plebian-model-sizer` provider for read-only resource
recommendations:

```sh
./kilix-tts --recommend
./kilix-stt --recommend --json
```

Press **n** in the read-aloud settings, or on dictation's Models tab, for the
same report; dictation then shows each model's resource verdict. Results are
advisory: selecting, saving defaults and installing stay explicit actions, and
`--recommend` refuses to combine with any of them. Exit 0 means a valid report,
which may have an empty shortlist; provider failures exit 1 and incompatible
options exit 2.

Install `plebian-model-sizer` 0.2.0 or newer, or set `PLEBIAN_MODEL_SIZER` to its
executable; a source checkout also finds the sibling `kilix-system-monitor`
launcher. Voice exchanges bounded JSON with it (15-second timeout, 1 MiB reply
limit) and fetches no model, opens no audio device and writes no settings.
Candidates are ranked by memory cost, not speech quality, and nothing is
selected or qualified by the report.
