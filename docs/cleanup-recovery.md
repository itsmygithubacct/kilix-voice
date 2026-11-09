# Cleanup recovery and accelerator quarantine

Reference for [Kilix Voice](../README.md).

## Recovering incomplete voice cleanup

The daemon registers each recorder, recognizer, synthesizer and player before
it can launch work or enter teardown. It keeps the owner until that resource's
close and any accelerator acknowledgement succeed. This includes a first
cleanup still in flight, a failed cleanup being retried, and read-aloud's
synthesis clients and sink worker. The process group is private; its leader
is retained until the final group signal is sealed, before reaping. On Linux
6.9 and later, a non-inherited pidfd targets the original process group even
if another reaper releases its leader's PID. An outside reap is logged and
keeps cleanup unproven; ESRCH alone does not prove cleanup.

On older kernels with `pidfd_open` (including Linux 5.3+ WSL2 kernels),
cleanup opens a pidfd for each group member, validates membership, rechecks
its retained leader after the scan, and signals only those descriptors.
Bounded rescans catch members created during cleanup; incomplete cleanup
remains owned. Acquisition uses a gated launcher so identity is checked
before even a short-lived command executes. A failed check leaves cleanup
unproven.

Only kernels without `pidfd_open` use checked numeric `killpg`. This legacy
backend retains a check-to-signal window if a competing reaper releases the
leader between the check and syscall. Use a pidfd-capable kernel and do not
install a competing child reaper in the daemon. Explicit managed capture uses
its supervisor's cooperative stop and authenticated cleanup protocol; it
retains unproven ownership if the supervisor cannot supply proof.

Descendants that leave the private group with `setsid()` or `setpgid()`
survive a normal stop, an abort and shutdown. Configure custom commands to
stay in their group. For existing escapees, identify and stop them through
the provider's own controls, verify that they have exited and released the
device, or reboot if their ownership cannot be established. A logged old PID
or group number is not authorization to signal a later process.

Pending cleanup refuses new dictation and read-aloud. Settlement and admission
retry with one shared 250 ms wait budget, and each owner has at most one
cleanup worker in flight. Successful resources are removed independently;
unproven owners are never evicted. Active playback still supports normal
replacement and barge-in; its owner remains registered throughout.

`kilix voice doctor` and the launcher's summary show `cleanup retained` with
turn, resource kind and age. `status.retained_cleanup` includes a first running
attempt with `state: in_flight`; an acknowledgement failure has
`state: quarantined`, `reason: acknowledgement-failed` and fixed recovery
instructions. `status.resource_owners` also lists active registered resources.
Status does not retry existing owners or open an audio device. Capture and
playback detail show only the executable, never its configured arguments.
`listening: false` describes the turn, and is not proof that a retained
recorder stopped.

Retry a request for another bounded cleanup attempt. If it persists, stop the
daemon normally and inspect stderr. Shutdown starts registered cleanups in
workers, spends one shared 5-second turn-join budget, then independently sends
the final SIGKILL to every still-owned process group, including first attempts.
It spends one shared additional 1-second cleanup wait and logs every unproven
resource by turn, kind and group. It never waits on a resource's close or
lifecycle lock on the shutdown thread. These are userspace wait bounds; a
kernel-stuck system call cannot be forced to return by Python.

A kernel-uninterruptible child may survive SIGKILL. Resolve its device/kernel
problem and verify exit, or reboot if recovery is impossible. Logged group
numbers are diagnostic; they do not authorize later signals after reuse.
Restart alone does not prove that the previous recorder or provider exited.

### Accelerator quarantine

The resource retains its original grant until teardown and acknowledgement
succeed. An acknowledgement-lock deadline keeps the descriptor and can retry.
Any acknowledgement failure that closes the descriptor is permanent: a later
no-op release cannot become proof. Status, doctor and shutdown keep reporting
that quarantine and accelerator admission remains closed.

To clear a permanent quarantine, first stop **all cooperating providers** and
prove all their owned jobs and descendants have exited. If that cannot be
established, reboot before recovery. With those providers kept stopped, an
administrator must remove/reset the shared namespace **and its sibling
anchor together**, then restart the cooperating providers against the same
fresh namespace. The default directory is
`/run/user/<effective-uid>/kilix-device-leases-v1`; its sibling anchor is
`.kilix-device-leases-v1.lease-v1.anchor`. For a configured private namespace,
the anchor is `.<directory-name>.lease-v1.anchor` in its parent directory.
Do not delete just the registry record or reset state while any provider may
still own work. Voice never performs this administrative reset. The external
lease component is unchanged.
