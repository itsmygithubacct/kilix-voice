"""kilix-voice's client of the one accelerator lease home (S04).

kilix-voice must never run an engine on an accelerator without a grant from
the shared execution lease, kilix.device-lease/v1, that the Qwen and
transcription providers also coordinate through. That lease has one home, the
kilix-device-lease component of kilix-system-monitor, and no lease code lives
here: this module imports that component, and turns its refusals into the
protocol's closed codes.

A CPU engine acquires nothing and imports nothing. A CPU runtime cannot claim
a GPU grant, and taking one would queue dictation behind Qwen jobs for no
hardware reason. Every engine kilix-voice ships today is a CPU engine.

Importing this module performs no filesystem work and imports no lease code.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import re
import time
from collections.abc import Callable, Iterator

from . import protocol, resources

LEASE_VERSION = "kilix.device-lease/v1"
LEASE_HOME = "kilix-system-monitor"
LEASE_DISTRIBUTION = "kilix-device-lease"
LEASE_MODULE = "kilix_device_lease"
# sha256 of contracts/kilix.device-lease-v1.interface.json at the home's
# commit this client was written against (5706b23).
LEASE_INTERFACE_SHA256 = "5e10f0c3c7096d7b1ca95835ab24924a6a3819fc2959f9403a146886083beaa8"

# The lease workload each engine task runs as.
WORKLOAD_FOR_TASK = {"transcribe": "stt-job", "translate": "stt-job",
                     "diarize": "stt-job", "synthesize": "tts-utterance"}
# The lease's own device label pattern; a label that does not match is a
# configuration mistake, refused before the lease is asked.
DEVICE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z", re.ASCII)
# The daemon-config key naming the accelerator a non-CPU engine runs on.
DEVICE_KEY = "stt.accelerator_device"

# kilix.device-lease/v1's refusal codes, as the protocol states them.
# invalid-request means this client asked wrongly: a daemon bug.
_CODE_FOR_LEASE_ERROR = {
    "queue-full": protocol.ERR_BUSY,
    "cancelled": protocol.ERR_CANCELLED,
    "deadline": protocol.ERR_DEADLINE,
    "unavailable": protocol.ERR_UNAVAILABLE,
    "lost-lease": protocol.ERR_UNAVAILABLE,
    "invalid-request": protocol.ERR_INTERNAL,
}

# The kilix-voice tree this package belongs to. A lease module loaded from
# inside it is a copy, and a copy is a second lease home.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class AcceleratorRefused(RuntimeError):
    """An engine was not given the accelerator; ``code`` says why, closed."""

    def __init__(self, message: str, code: str) -> None:
        if code not in protocol.ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code


def _module():
    """The lease home's client module, or AcceleratorRefused (unavailable)."""
    try:
        module = importlib.import_module(LEASE_MODULE)
    except ImportError as error:
        raise AcceleratorRefused(
            f"the accelerator lease client ({LEASE_MODULE}) is not installed, so "
            "an engine that needs an accelerator cannot run. Install the "
            f"{LEASE_DISTRIBUTION} distribution from {LEASE_HOME} into the "
            "interpreter kilix-voice runs with.", protocol.ERR_UNAVAILABLE) from error
    if getattr(module, "VERSION", None) != LEASE_VERSION:
        raise AcceleratorRefused(
            f"the installed {LEASE_MODULE} speaks "
            f"{getattr(module, 'VERSION', None)!r}; kilix-voice needs "
            f"{LEASE_VERSION}. Install the matching {LEASE_DISTRIBUTION}.",
            protocol.ERR_UNAVAILABLE)
    where = os.path.realpath(getattr(module, "__file__", None) or "")
    if not where or where.startswith(_PACKAGE_ROOT + os.sep):
        raise AcceleratorRefused(
            f"{LEASE_MODULE} was loaded from inside kilix-voice ({where or 'nowhere'}). "
            f"The lease has one home, {LEASE_HOME}; remove the copy and install "
            f"{LEASE_DISTRIBUTION}.", protocol.ERR_UNAVAILABLE)
    return module


class ExecutionGrant:
    """A held grant. Hand ``guard_fd`` to any engine process; call reaped() after teardown."""

    def __init__(self, lease, refused: Callable[[Exception], AcceleratorRefused]) -> None:
        self._lease = lease
        self._refused = refused
        self._reaped = False

    @property
    def guard_fd(self) -> int:
        try:
            return self._lease.guard_fd
        except Exception as error:
            raise self._refused(error) from error

    def reaped(self) -> None:
        """Record that the engine closed and every process it started is gone."""
        self._reaped = True


@contextlib.contextmanager
def execution(*, device_class: str, task: str, job_id: str,
              deadline_monotonic: float | None, token,
              peer_gone: Callable[[], bool] | None = None,
              progress: Callable[[dict], None] | None = None,
              device_label: str = "",
              namespace: str | None = None) -> Iterator[ExecutionGrant | None]:
    """Hold the accelerator lease for one engine run; None for a CPU engine.

    ``token`` is the job's cancellation token: a stop refuses a wait with
    `cancelled`, and ``deadline_monotonic`` bounds it with `deadline`.
    ``peer_gone`` ends a wait nobody is left to deliver to. ``progress`` is
    told {state, position} while the request waits in the queue.
    ``namespace`` is for isolated tests only; the daemon never passes one, so
    it always uses the one namespace every provider shares.

    On exit the grant is released. It is released as cleaned up only when
    reaped() was called; otherwise the lease quarantines the accelerator,
    because an engine whose teardown was not proven may still hold it.
    Raises AcceleratorRefused with a closed code.
    """
    if device_class == resources.DEVICE_CPU:
        yield None
        return
    if device_class not in resources.DEVICE_CLASSES:
        raise AcceleratorRefused(
            f"device class {device_class!r} is not one kilix-voice knows; the "
            "engine was not started.", protocol.ERR_UNAVAILABLE)
    workload = WORKLOAD_FOR_TASK.get(task)
    if workload is None:
        raise AcceleratorRefused(
            f"an engine task {task!r} has no accelerator workload; the engine "
            "was not started.", protocol.ERR_UNAVAILABLE)
    if not isinstance(device_label, str) or not DEVICE_LABEL.fullmatch(device_label):
        raise AcceleratorRefused(
            f"this engine runs on a {device_class} accelerator, and {DEVICE_KEY} "
            f"{'is not set' if not device_label else 'is not a device label'}. "
            "Set it, in the daemon config, to the label the accelerator lease "
            "knows the device by.", protocol.ERR_UNAVAILABLE)
    module = _module()

    def refused(error: Exception) -> AcceleratorRefused:
        code = _CODE_FOR_LEASE_ERROR.get(getattr(error, "code", None), protocol.ERR_INTERNAL)
        return AcceleratorRefused(
            f"the accelerator lease refused the engine ({getattr(error, 'code', type(error).__name__)}): "
            f"{error}", code)

    ceiling = time.monotonic() + float(module.MAX_WAIT_SECONDS)
    deadline = ceiling if deadline_monotonic is None else min(deadline_monotonic, ceiling)
    report = None
    if progress is not None:
        def report(status) -> None:
            progress({"state": status.state, "position": status.position})
    try:
        lease = module.acquire(job_id=job_id, workload=workload, device=device_label,
                               deadline=deadline, cancelled=token.is_set,
                               disconnected=peer_gone, progress=report,
                               namespace=namespace)
    except module.LeaseError as error:
        raise refused(error) from error
    grant = ExecutionGrant(lease, refused)
    failed = False
    try:
        yield grant
    except BaseException:
        failed = True
        raise
    finally:
        try:
            lease.release(cleanup_complete=grant._reaped)
        except module.LeaseError as error:
            if not failed:
                raise refused(error) from error
