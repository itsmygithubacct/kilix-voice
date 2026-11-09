# Safety model and model licences

Reference for [Kilix Voice](../README.md).

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
| `whisper-small-en` | `faster-whisper-small-en` |

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
`kilix-tts --install` for the Piper voice. `kilix-stt --install
whisper-small-en` hands straight to `kilix models install`: that command is
the acceptance route itself, so it is not asked for a receipt first.

One coupling is outside this repository: at the pinned ref,
`kilix/scripts/install-kilix-voice.sh` fetches the Vosk wheel and the model in
one block, and its `--without-dictation` flag skips both. So refusing the
weights on that leg also withholds the library. Keeping the library installable
on its own needs a library-only leg in that installer; nothing here can do it.
