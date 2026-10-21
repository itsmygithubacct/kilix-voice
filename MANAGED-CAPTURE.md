# Explicit managed capture

`MicCapture(cfg)` retains the published Whistle/daemon capture behavior.
`MicCapture(cfg, control=CaptureControl(...))` selects managed capture for
Transcribe and Voicebox: bounded recording, explicit lock/cancel/disconnect
authority, a dedicated child supervisor, exact credentialed cleanup proof and
persistent quarantine on unproved cleanup. Construction opens no audio device.

The managed owner/supervisor and capture implementation are carried from
`869366f22bb930230778b7cec5229f42b4211131`, including its allocation/startup
cleanup fixes after `c4431563b239e23b314be37b9ee96fbcb9a21b7d`. This commit
descends from the published Whistle source `cba6048cc7d8b6f33a1af5cc3cf84e5c1c34e802`;
it does not replace that history with the divergent managed fixture branch.

`voicelib.device_leases` is a compatibility adapter to the canonical
`kilix_device_lease` component of kilix-system-monitor. It carries no registry
implementation. Install that component outside the Voice package in the same
interpreter closure. All cooperating providers use its existing namespace,
grant, cleanup acknowledgement and quarantine state. Missing, wrong-version or
in-package copies refuse. Default Voice accelerator behavior is preserved.

Private host generations accept explicit Voice/lease source commits and bind
both complete populations. Current carrier/provider pins are unchanged until
the owner chooses a reviewed integration tuple. Synthetic capture/source tests
do not qualify physical microphones, installed guests or model fit.
