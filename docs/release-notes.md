# Release notes

Reference for [Kilix Voice](../README.md).

## Release history

For foreground prompt/play/replay sessions, offline resident Qwen auditions,
and first-use Qwen downloads, see [Interactive TTS](interactive-tts.md).
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
