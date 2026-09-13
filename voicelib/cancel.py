"""One cancellation token per turn: explicit stop and caller budget together.

R3 F01 and F02 were both the same defect wearing two hats.  Speech carried a
deadline that generation checked and playback did not; dictation carried one
that nothing read at all.  Each check site owned its own notion of "should I
stop", so a correct check in one place did not make the turn stop in another.

This module makes the notion a single object.  A turn holds one `Cancellation`.
Anything that can block takes its remaining budget from that object, and
anything that can stop asks that object whether to.  There is no second flag to
fall out of step with.

It is deliberately `threading.Event`-compatible -- `set`, `is_set`, `wait`,
`clear` -- so a turn's `.cancel` attribute keeps behaving as the Event it
replaces, and every existing call site keeps its meaning.  What is new is
`expired`, `remaining`, `done`, `wait_done` and `budget`.
"""

from __future__ import annotations

import threading
import time

__all__ = ["Cancellation", "Cancelled", "DeadlineExceeded", "CancellationError"]


class CancellationError(RuntimeError):
    """Base for a turn that must not continue."""


class Cancelled(CancellationError):
    """The turn was explicitly stopped."""


class DeadlineExceeded(CancellationError):
    """The caller's budget ran out."""


class Cancellation:
    """Explicit cancel plus an optional absolute deadline, as one token.

    `deadline` is an absolute MONOTONIC instant, not a duration.  A duration
    would have to be re-based at every check, and a wall-clock instant would
    move under an NTP step mid-turn.  `None` means the caller set no bound.

    `clock` is injectable so tests can drive expiry without sleeping.
    """

    __slots__ = ("_event", "deadline", "_clock")

    def __init__(self, deadline: float | None = None, clock=time.monotonic) -> None:
        self._event = threading.Event()
        self.deadline = deadline
        self._clock = clock

    # -- threading.Event surface (unchanged meaning: EXPLICIT cancel only) ---

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def clear(self) -> None:
        self._event.clear()

    def wait(self, timeout: float | None = None) -> bool:
        """Block for an explicit cancel.  Deliberately deadline-blind.

        Call sites that meant "has someone pressed stop" keep that meaning.
        Call sites that meant "should I still be running" want `wait_done`.
        """
        return self._event.wait(timeout)

    # -- the budget half ----------------------------------------------------

    def expired(self) -> bool:
        """True once the caller's budget has run out."""
        return self.deadline is not None and self._clock() >= self.deadline

    def remaining(self) -> float | None:
        """Seconds of caller budget left; None when unbounded, never negative."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - self._clock())

    def done(self) -> bool:
        """True when the turn must stop, for either reason."""
        return self._event.is_set() or self.expired()

    def reason(self) -> str | None:
        """Why the turn must stop.  Explicit cancel outranks expiry.

        Ordering matters: a turn that was stopped and then sat past its
        deadline was stopped, and reporting a deadline would misattribute it.
        """
        if self._event.is_set():
            return "cancelled"
        if self.expired():
            return "deadline"
        return None

    def check(self) -> None:
        """Raise if the turn must stop.  Use at boundaries inside blocking work."""
        why = self.reason()
        if why == "cancelled":
            raise Cancelled("the turn was stopped")
        if why == "deadline":
            raise DeadlineExceeded("the request deadline elapsed")

    def wait_done(self, timeout: float | None = None) -> bool:
        """Block until cancelled, expired, or `timeout`.  True if it must stop.

        The wait is clipped to the remaining budget, so expiry wakes the caller
        AT the deadline rather than up to one poll interval past it.  That is
        the difference between a deadline the code honours and one it rounds.
        """
        left = self.remaining()
        if left is not None:
            timeout = left if timeout is None else min(timeout, left)
        if timeout is not None and timeout <= 0:
            return self.done()
        self._event.wait(timeout)
        return self.done()

    def budget(self, cap: float) -> float:
        """The timeout to hand a blocking call that would otherwise take `cap`.

        Never longer than the caller's remaining budget, so a 210 s engine
        timeout cannot outlive a 2 s request.  Raises if there is nothing left,
        because starting unbounded work with an exhausted budget is the bug
        F01 and F02 both described.
        """
        self.check()
        left = self.remaining()
        return cap if left is None else min(cap, left)
