# Desktop integration

Reference for [Kilix Voice](../README.md).

## System voice volume

System announcements use a separate output gain capped at 20% (the default).
`kilix tts --system-voice --system-volume=10` selects a quieter session;
`--system-volume=0` mutes it. The matching Kilix 95 System Voice dialog saves
this choice privately. Greetings, cached health/power notices and goodbye
all pass through PCM attenuation before the output backend, so the cap applies
with every supported audio player. This does not change the master mixer or
ordinary read-aloud, document-reader or media-player volume.

## Desktop startup voice

Desktop startup voice uses `kilix tts --prepare-system-voice` in a terminal for
explicit first-use setup, then `kilix tts --system-voice --speak hello` with a
pipe held open by the desktop. The latter is offline: it requires installed
weights and their covering receipt, speaks once, and keeps Piper warm until
stdin closes or the desktop terminates it. `--speak ''` preloads silently.

## Document reading and Kristin opt-in

Kilix 95 provides **Start → Programs → Document Reader** for TXT, Markdown and
PDF files, with passage progress, Play, Pause/Resume, Stop, Previous and Next.
It owns playback independently of the voice daemon. Pause resumes at the start
of the current passage. PDFs require `pdftotext` (`poppler-utils`); scanned pages
need OCR. Input is bounded to 16 MiB per file and 8 MiB of extracted UTF-8 text.

To explicitly install Kristin and choose it as the shared read-aloud default:

```sh
kilix tts --enable-kristin
```

This requires a terminal, presents the voice download notice, and saves the
preference only after installation succeeds. It does not enable startup speech.
The desktop's **Use Kristin for read-aloud…** button runs this same opt-in.

`--reader-session FILE` is the desktop bridge: newline-delimited JSON status on
stdout, with private single-byte stdin commands (`p` play/resume, `a` pause,
`s` stop/reset, `n` next, `b` previous, `q` quit). EOF stops its owned playback.
Opening a session only loads text; speech starts on `p`. The current shared
voice is read for each passage. No model is downloaded by opening a document.

## Avatar-owned turns

The optional `owned-speech/v1` and `owned-dictation/v1` capabilities add
owner-scoped speech status/stop and press-to-talk recording. Speech utterance
IDs are idempotent, bounded to 128 retained results, and expire after five seconds
without an owner heartbeat. Microphone turns expire after three seconds without
a heartbeat. Releasing press-to-talk closes capture immediately and drains the
buffer before producing the final transcript. Continuous input uses a per-turn
silence interval, defaulting to three seconds. These operations retain the
normal consent, resource, deadline, cancellation and job-outcome checks.

Owned speech never replaces other audio: `owned-speak` is refused as `busy`
while anything is speaking or listening, including a legacy `speak`. Preserve
the returned `turn` (`speak-<daemon instance>-<n>`); `owned-stop` needs it and
the same owner, so a stale stop — including one kept across a daemon restart —
cannot silence a successor. Poll `owned-status` at least once per second: it
renews the lease, while global `status` does not. It reports `preparing`,
`speaking`, `completed`, `cancelled` (stopped by `owned-stop` or `stop-speech`),
`superseded` (a legacy `speak` replaced it), `expired` (the lease or request
deadline ran out), `failed` or `unknown`, with any reason in `detail`.
`playing` describes the active playback process, not sample-accurate timing.

Repeating an identical `owned-speak` while its result is retained returns that
result without replaying audio; reusing an utterance token for different text
or options is refused as `malformed`. After a lost acknowledgement, reconcile
through `owned-status`. Never replay an `unknown` utterance automatically: the
result may have aged out or the daemon may have restarted. Owner tokens
coordinate same-user clients; they are not protection against other code
running under that UID.

`owned-speech-volume/v1` adds an optional integer `volume` (0–100) to
`owned-speak`. It attaches gain to that turn only. Send `owned-volume` with the
same `owner`, current `turn`, and `volume` to adjust speech already playing;
the reply's `applied` is false for a stale or mismatched turn. These updates
renew the speech lease. They never change another turn or the system mixer.
Repeating an utterance still uses its original admission options, even after
its playback volume changes.

Conversation gain is applied to small PCM blocks before every supported output
backend, without resynthesis. Standard output queues are kept short; custom
sinks may buffer longer. Unscoped speech keeps its existing gain. Avatar's
0–100% preference is separate from the system-announcement path, which retains
its configurable 0–20% cap, and from intro/outro music.
