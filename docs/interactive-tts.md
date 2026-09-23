# Interactive TTS auditions

`kilix-tts --interactive --model espeak` opens a foreground prompt/play loop.
It neither changes read-aloud settings nor requires `kilix-voiced`. Playback
uses the local machine's default audio sink, including when launched over SSH.
The tool keeps the selected engine resident between lines. `/help` lists voice
and language selection, replay, exclusive WAV saving and exit commands. Ctrl-C
cancels the current operation; Ctrl-D exits. The tool writes no prompt history,
but the hosting terminal's normal transcript/scrollback policy still applies.

## Resident Qwen test runtime

Use an isolated Python environment containing upstream `qwen-tts` and its
dependencies. Invoke the CLI with that environment's Python:

```sh
/path/to/venv/bin/python /path/to/kilix-tts --interactive \
  --qwen-model-dir /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --voice Ryan --language English --device cpu --threads 4
```

This explicitly selected development audition uses existing local weights
only: loading never downloads weights, never enables remote model code, and
does not select or qualify a production provider. CPU/float32 is the default;
`--device cuda:0` explicitly requests CUDA/bfloat16. Check available GPU memory
before doing so. There is no automatic CPU/GPU fallback. Qwen's natural rate
does not accept `--rate`; `--seed` controls sampling, not a reproducibility
guarantee. Generation is bounded to approximately 60 seconds of audio per line,
and text is limited to 16 KiB of UTF-8. CPU inference can be much slower than
playback. Whole clips play after synthesis; this mode is not streaming.

For FlashAttention 2, use an environment with CUDA-enabled PyTorch and a
compatible `flash-attn` build, then select both flags explicitly:

```sh
/path/to/gpu-venv/bin/python /path/to/kilix-tts --interactive \
  --download-qwen qwen3-tts-0.6b-customvoice --voice Ryan \
  --device cuda:0 --attention flash_attention_2
```

The runtime checks that CUDA, an Ampere-or-newer GPU, and the extension are
available, then passes `flash_attention_2` to Qwen with bfloat16 weights. It
does not silently fall back to SDPA. The default remains `--attention sdpa`
for CPU auditions. Installing `flash-attn` into a CPU-only PyTorch environment
does not enable GPU inference; use a separate CUDA environment if needed.

For an existing **Base** model, supply `--synthetic-reference`, without
`--voice`. The tool makes a temporary eSpeak reference and caches its Qwen
conditioning in memory. This tests Base synthesis without cloning a person's
voice; its timbre is not representative of CustomVoice presets. VoiceDesign
is downloadable below but is not yet supported by the interactive session.

Add `--speak "Hello from Kilix."` to run one greeting before the prompt.
`/save example.wav` writes the most recent completed clip and refuses an
existing file or symlink. A cancelled generation keeps the previous good clip.

## Downloading catalogued Qwen models

Install the 0.2.2 `kilix-content` first-use implementation and `kilix-license`
in the same environment, then run:

```sh
/path/to/venv/bin/python /path/to/kilix-tts --download-qwen all
```

`all` means the three models currently in the determined Content catalog:
0.6B CustomVoice, 0.6B Base, and 1.7B VoiceDesign. It does **not** mean all five
upstream variants. 1.7B Base and 1.7B CustomVoice need catalog records before
this download action can offer them; existing local Base/CustomVoice weights
can still be auditioned explicitly.

The command displays the authority's verbatim licence screen, explains that
continuing accepts it, then asks **Press q to quit or any key to continue**.
No Enter or typed agreement sentence is required. `q`, `Q`, Ctrl-C, Ctrl-D
and end-of-input do not accept. The terminal mode is restored after the key
or an interrupt, and stale pending input cannot accept a new licence.
Quitting writes no receipt and downloads no weights. Existing covering
receipts are reused. Content downloads
from its pinned upstream URLs, checks the catalog hashes, and publishes the
complete asset atomically. Receipts use the authority's shared store. Weights
go under `$GPU_TERMINAL_HOME/tts-auditions/content/assets/MODEL/model`, defaulting
to `~/.local/gpu_terminal/tts-auditions/content/assets/MODEL/model`. The final
printed path can be passed to `--qwen-model-dir`. No system defaults change.

To go directly from first-use setup into a warm interactive speech session in
the **same terminal**, select one model:

```sh
/path/to/venv/bin/python /path/to/kilix-tts --interactive \
  --download-qwen qwen3-tts-0.6b-customvoice --voice Ryan
```

It verifies an existing installation or asks for the continuation key and
downloads it, then opens `tts>`. `q` exits without starting synthesis.
For `qwen3-tts-0.6b-base`, add `--synthetic-reference` instead of `--voice`.
`all` and VoiceDesign remain download-only options, not interactive engines.

## Hardware-aware audition tiers

`kilix-tts --tiers` lists `minimal` (eSpeak), `small` (MBROLA us1),
`neural` (Piper Kristin medium), `qwen-cpu` (0.6B CustomVoice float32/SDPA),
and `qwen-gpu` (0.6B CustomVoice bfloat16/FlashAttention 2).
Add `--json` for evidence, memory budgets and availability reasons.

Use `kilix-tts --interactive --tier neural` to select one. Each selection
checks current headroom again. Listing loads no weights. If Piper's runtime is
available but its model is missing, the selection shows the full licence and
waits for a fresh keypress before the checksum-pinned installation. Press `q`
to leave without installing. On the installed Kilix desktop, `kilix tts`
installs the pinned Piper runtime only on first Piper selection. On x86_64
Debian, `kilix tts --interactive --tier qwen-cpu` similarly installs the locked
CPU runtime on first selection; if weights are missing, it then presents their
licence and downloads them only after a keypress. This can use several GiB of
disk and CPU synthesis may take about a minute for a short sentence. Unavailable
tiers remain visible. Direct `kilix-tts` Qwen use still requires its **current
Python interpreter** to contain the Qwen dependencies. The GPU tier requires
an independently provisioned CUDA/FlashAttention runtime. For the installed
Kilix command, set `KILIX_QWEN_GPU_PYTHON` to that environment's absolute
Python path before selecting `qwen-gpu`; Kilix still verifies fit and runtime
availability before offering missing weights. The tier also requires
physical GPU 0 without CUDA device remapping; it refuses a budget for another
GPU. Explicit `--qwen-model-dir` remains an expert path outside tier gating.

Install the shared `plebian-model-sizer` with the dated audition profiles, or
set `PLEBIAN_MODEL_SIZER` to its executable. Kilix Voice delegates fit math
to that provider: reference peak RAM/VRAM plus 20%, with 256 MiB kept free in
each memory pool. Unknown headroom and unsupported hardware cannot become a
selectable tier. Missing Qwen or Piper weights are offered for first-use only
when their runtime and memory fit. Disk acquisition costs remain unknown;
installation still needs explicit interactive selection and licence acceptance.

These development profiles measure one short sentence, not maximum prompt
length or speech quality. Piper's measurement uses its direct engine API;
the interactive provider path adds process/IPC overhead. Qwen CPU fits some
machines' memory but is much slower than real time. Only one tier is planned
at a time; longer prompts and concurrent workloads need additional headroom.
