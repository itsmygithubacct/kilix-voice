"""The daemon resource contract, shared by every registered resource kind.

starting -> ready -> in use; close permanently refuses new operations and
enters closing. Only zero starters/users and successful disposal prove closed.
A deadline, a failed disposal, or a busy handoff leaves unproven ownership.
Allocation is counted before it begins and published before the count drops.
Close never frees a user's handles, and only one disposal may run, including
constructor failure cleanup. The daemon retains unproven owners and escalates
their GroupPopen directly; a signal or a handoff is never reap proof.
"""
import functools
import threading
import time


CLOSE_WAIT_S = 0.1
RESOURCE_KINDS = {}
_creation_lock = threading.Lock()


class Lifecycle:
    def __init__(self):
        self.condition = threading.Condition()
        self.starting = 0
        self.users = 0
        self.threads = {}
        self.requested = False
        self.disposing = False
        self.state = "ready"

    def enter(self, owner, starting=False):
        ident = threading.get_ident()
        with self.condition:
            if self.requested and (starting or not self.threads.get(ident)):
                raise owner._closed_error(owner.closed_message)
            self.threads[ident] = self.threads.get(ident, 0) + 1
            if starting:
                self.starting += 1
            else:
                self.users += 1
            self.state = "starting" if self.starting else "in use"

    def leave(self, starting=False):
        ident = threading.get_ident()
        with self.condition:
            if starting:
                self.starting -= 1
            else:
                self.users -= 1
            self.threads[ident] -= 1
            if not self.threads[ident]:
                del self.threads[ident]
            if not self.requested:
                self.state = "starting" if self.starting else "in use" if self.users else "ready"
            self.condition.notify_all()

    def close(self, owner, dispose):
        deadline = time.monotonic() + CLOSE_WAIT_S
        with self.condition:
            self.requested = True
            if self.state == "closed" or self.disposing:
                return
            self.disposing = True
            self.state = "closing"
        proven = False
        try:
            # This can signal, but must not free native handles or reap a
            # decoder's child. Shutdown can independently signal its handle.
            owner._on_close_requested()
            with self.condition:
                while self.starting or self.users:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return
                    self.condition.wait(left)
            dispose()
            proven = owner._dispose_complete()
        finally:
            with self.condition:
                self.state = "closed" if proven else "unproven"
                self.disposing = False
                self.condition.notify_all()


class Resource:
    lifecycle_error = RuntimeError
    closed_message = "resource is closing or closed"
    cleanup_method = "close"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for kind in cls.__dict__.get("resource_kinds", ()):
            if kind in RESOURCE_KINDS:
                raise TypeError("duplicate resource kind: " + kind)
            RESOURCE_KINDS[kind] = cls

    @property
    def lifecycle(self):
        # Also supports inert fixtures constructed with object.__new__.
        with _creation_lock:
            if "_resource_lifecycle" not in self.__dict__:
                self._resource_lifecycle = Lifecycle()
            return self._resource_lifecycle

    @property
    def cleanup_complete(self):
        life = self.lifecycle
        with life.condition:
            return (life.state == "closed" and not life.starting
                    and not life.users and not life.disposing)

    def _on_close_requested(self):
        pass

    def request_close(self):
        """Refuse further allocation even while a constructor owns cleanup."""
        life = self.lifecycle
        with life.condition:
            life.requested = True
            if life.state != "closed":
                life.state = "closing" if life.disposing else "unproven"

    def _dispose_complete(self):
        return True

    def _closed_error(self, message):
        error = self.lifecycle_error(message)
        error.resource_closed = True
        return error

    def _cancel_published_process(self, process):
        from .processes import kill_group
        kill_group(process)

    def escalate_cleanup(self):
        process = getattr(self, "_process", None)
        if process is not None:
            self._cancel_published_process(process)

    def _publish(self, field, allocate):
        """Publish before releasing startup ownership, even after cancellation.

        No condition is held across an allocator: a stuck allocator must not
        prevent a bounded close or a read-only ownership diagnostic.
        """
        life = self.lifecycle
        life.enter(self, starting=True)
        try:
            value = allocate()
            setattr(self, field, value)
            if life.requested:
                # Allocation may have been inside a syscall when close won.
                # Publish first, then refuse the handoff. A late child must
                # be interrupted even if shutdown's escalation pass is over.
                if field == "_process":
                    self._cancel_published_process(value)
                raise self._closed_error("resource closed during allocation")
            return value
        finally:
            life.leave(starting=True)


def starting(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        life = self.lifecycle
        life.enter(self, starting=True)
        try:
            result = function(self, *args, **kwargs)
        finally:
            life.leave(starting=True)
            if life.requested and not life.starting and not life.users:
                getattr(self, self.cleanup_method)()
        if life.requested:
            raise self._closed_error("resource closed during startup")
        return result
    return wrapped


def constructing(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return starting(function)(self, *args, **kwargs)
        except BaseException:
            if getattr(self, "_owner_ready", False):
                getattr(self, self.cleanup_method)()
            raise
    return wrapped


def using(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        life = self.lifecycle
        life.enter(self)
        try:
            return function(self, *args, **kwargs)
        finally:
            life.leave()
    return wrapped


def closing(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        return self.lifecycle.close(self, lambda: function(self, *args, **kwargs))
    return wrapped
