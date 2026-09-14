"""Exactly one terminal outcome per job, recorded once, retrievable by id.

P07 and P11. Every job the daemon accepts -- a speak with at least one clip, a
dictate, an ingest -- ends in exactly one outcome from a closed set. The
outcome is the single source for every channel that reports how a job ended:
status, the terminal message a protocol-1.2 chunk subscriber receives, and
the dictation terminal datagram. Those channels cannot disagree because none
of them decides the outcome for itself. A refused request is not a job; its
reply is its only outcome.

Exactly-once is enforced on the TURN, which settles first-writer-wins under
its own lock. This ledger is only where settled outcomes are kept, so a daemon
built without __init__ -- as tests and review harnesses build them -- still
settles its turns correctly.

The ledger holds prose failure messages only, never spoken or recognised
text: that text is the user's, and status is readable by any process of theirs.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict

from . import protocol

# The protocol's vocabulary, so the wire and the ledger share one list.
OUTCOMES = protocol.JOB_OUTCOMES
OUTCOME_COMPLETED, OUTCOME_CANCELLED, OUTCOME_DEADLINE, OUTCOME_FAILED = OUTCOMES

KIND_SPEECH = "speech"
KIND_DICTATION = "dictation"
KIND_INGEST = "ingest"
KINDS = (KIND_SPEECH, KIND_DICTATION, KIND_INGEST)

MAX_ENTRIES = 64
RECENT = 16
MAX_MESSAGE_CHARS = 512

# The code an outcome other than `failed` must carry. `failed` carries any
# other closed code.
_FIXED_CODES = {OUTCOME_COMPLETED: None, OUTCOME_CANCELLED: protocol.ERR_CANCELLED,
                OUTCOME_DEADLINE: protocol.ERR_DEADLINE}


@dataclasses.dataclass(frozen=True)
class JobOutcome:
    """How one job ended. Its code is consistent with its outcome, always."""

    job: str
    kind: str
    outcome: str
    code: str | None
    message: str = ""
    chunks_published: int = 0
    subscriber_lost: bool = False
    delivered: bool | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.job, str) or not self.job
                or len(self.job) > protocol.MAX_ID_CHARS):
            raise ValueError(f"job id must be 1-{protocol.MAX_ID_CHARS} characters")
        if self.kind not in KINDS:
            raise ValueError(f"unknown job kind {self.kind!r}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}")
        if self.outcome in _FIXED_CODES:
            if self.code != _FIXED_CODES[self.outcome]:
                raise ValueError(
                    f"outcome {self.outcome!r} must carry code "
                    f"{_FIXED_CODES[self.outcome]!r}, not {self.code!r}")
        elif (self.code not in protocol.ERROR_CODES
              or self.code in (protocol.ERR_CANCELLED, protocol.ERR_DEADLINE)):
            raise ValueError(
                f"a failed job carries a failure code, not {self.code!r}")
        if not isinstance(self.message, str):
            raise ValueError("an outcome message is prose")
        if (not isinstance(self.chunks_published, int)
                or isinstance(self.chunks_published, bool)
                or self.chunks_published < 0):
            raise ValueError("chunks_published is a count")
        if not isinstance(self.subscriber_lost, bool):
            raise ValueError("subscriber_lost is true or false")
        if self.delivered is not None and not isinstance(self.delivered, bool):
            raise ValueError("delivered is true, false or unknown")
        if len(self.message) > MAX_MESSAGE_CHARS:
            object.__setattr__(self, "message",
                               self.message[:MAX_MESSAGE_CHARS] + " …[truncated]")

    def record(self) -> dict:
        """This outcome as status reports it."""
        out: dict = {"job": self.job, "kind": self.kind, "state": "settled",
                     "outcome": self.outcome, "chunks": self.chunks_published}
        if self.code is not None:
            out["code"] = self.code
        if self.message:
            out["error"] = self.message
        if self.subscriber_lost:
            out["subscriber_lost"] = True
        if self.delivered is not None:
            out["delivered"] = self.delivered
        return out


class JobLedger:
    """Where settled outcomes are kept, newest last, bounded.

    ``begin`` records a job as running, but only if nothing is recorded for
    it: a worker that settled before its handler got round to begin must
    not be put back to running. ``record`` refuses a second outcome for the
    same job and changes nothing. Only settled entries are evicted, so a job
    still running is always found.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[str, str] | JobOutcome] = OrderedDict()
        self._max = max_entries

    def begin(self, job: str, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        with self._lock:
            if job not in self._entries:
                self._entries[job] = ("running", kind)
                self._evict_locked()

    def record(self, outcome: JobOutcome) -> bool:
        with self._lock:
            if isinstance(self._entries.get(outcome.job), JobOutcome):
                return False
            self._entries[outcome.job] = outcome
            self._evict_locked()
            return True

    def mark_delivered(self, job: str, delivered: bool) -> None:
        """Record whether a settled job's terminal message reached its receiver."""
        with self._lock:
            entry = self._entries.get(job)
            if isinstance(entry, JobOutcome) and entry.delivered is None:
                self._entries[job] = dataclasses.replace(entry, delivered=bool(delivered))

    def get(self, job: str) -> dict | None:
        with self._lock:
            entry = self._entries.get(job)
        return None if entry is None else self._describe(job, entry)

    def recent(self, n: int = RECENT) -> list[dict]:
        with self._lock:
            items = list(self._entries.items())[-n:]
        return [self._describe(job, entry) for job, entry in items]

    @staticmethod
    def _describe(job: str, entry) -> dict:
        if isinstance(entry, JobOutcome):
            return entry.record()
        _state, kind = entry
        return {"job": job, "kind": kind, "state": "running"}

    def _evict_locked(self) -> None:
        excess = len(self._entries) - self._max
        if excess <= 0:
            return
        for job in [j for j, e in self._entries.items() if isinstance(e, JobOutcome)][:excess]:
            del self._entries[job]
