# Pocket TTS CPU audition (development only)

Pocket TTS is an optional, foreground English TTS path using the upstream
Python API on CPU. It is not the native GGUF/Vulkan experiment, does not alter
the voice daemon or default model, and is not yet a qualified 0.2.2 tier.

The runtime is pinned separately in Kilix's `runtimes/pocket-cpu/uv.lock`.
Select it explicitly with `scripts/install-kilix-pocket-tts-cpu.sh`; nothing
is installed by browsing tiers. Run `kilix-tts` with the Python printed by
that installer, with Voice's source directory on `PYTHONPATH` or the installed
Voice package available in that interpreter.

The model directory is an already admitted, local copy of the pinned
`kyutai/pocket-tts-without-voice-cloning` revision
`d29db7978e464fb90cb3359ee0c69a273b9142cc`. It must contain:

```
model.safetensors
tokenizer.model
embeddings/alba.safetensors
```

The adapter checks byte counts and SHA-256 of all three files before loading,
uses the package's `english_2026-04` architecture with only local paths, and
sets Hugging Face offline mode. It supports only the Alba preset embedding;
arbitrary prompt audio, voice cloning, URL paths, and other presets are not
exposed. The `--pocket-model-dir` expert option does not grant a licence or
fetch files. Model acquisition must run through Kilix Content's first-use
licence receipt and explicit prohibited-use agreement before this path can be
made a release-selectable tier. Do not use the old `pocket-tts-english-q8_0`
conversion asset here: it lacks the Alba embedding and invokes a separate
native conversion tool.

After that admission, the session form is:

```
PYTHONPATH=/path/to/voice /path/to/pocket-python /path/to/voice/kilix-tts \
  --interactive --pocket-model-dir /path/to/admitted/model --voice Alba
```

Alba is voiced by Alba MacKenna. Preserve the upstream Pocket model's CC BY
4.0 attribution and prohibited-use notice, and the voice provenance when
shipping or sharing output. Before a 0.2.2 tier can be declared ready, test
speech quality, time to first reply, peak RAM, and the exact first-use receipt
for the new Python asset on target hardware.
