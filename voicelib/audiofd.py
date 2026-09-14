"""Audio crossing the process boundary as a descriptor, never as JSON.

A07 keeps audio out of JSON frames, and A08 and A15 still need playable audio
to reach a caller. A sealed memfd does both. The bytes live in an anonymous
file, the seals make that file immutable, and the caller receives a read-only,
close-on-exec descriptor by SCM_RIGHTS. Nothing is written to a filesystem, so
there is no path to validate, nothing to clean up and no lifetime to police.

The writer is ported from voicelib/providers.py::_new_audio_fd and _audio_fd
on the work/0.2.2-voice-snapshot-fd lineage, including its fix that closes
every descriptor when a hand-off fails. That lineage deletes Piper and the
mbrola fallback, so the function is carried over; the branch is not merged.

Linux only, like the daemon, which already needs SO_PEERCRED.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import select
import stat
import threading
import time

from . import protocol

# At most this many SCM_RIGHTS descriptors are received with one control
# request. More are closed by the kernel as the message is truncated.
MAX_INBOUND_FDS = 4

# A pipe must reach end-of-file within this long. Ingestion answers on the
# control connection, so this bounds how long a slow writer can hold the
# accept loop -- the same order as a peer that connects and says nothing.
INGEST_READ_TIMEOUT_S = 1.0
_READ_BLOCK = 1 << 20
_PIPE_BLOCK = 1 << 16
# Every step that touches a caller's descriptor runs on a helper thread. The
# accept loop waits for it no longer than the read's own bound plus this grace.
_READ_GRACE_S = 0.25
# A read still running when that wait ends is left to finish on its own
# thread, holding its own duplicate of the descriptor. At most this many are
# left running at once. Beyond that, a descriptor is refused busy before it is
# even inspected, so a caller cannot pile up stuck threads.
MAX_STALLED_READS = 2
_O_PATH = getattr(os, "O_PATH", 0o10000000)
_stalled_lock = threading.Lock()
_stalled = 0

_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
# The Linux UAPI values, used when this Python build does not name them.
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
ALL_SEALS = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL


class DescriptorError(ValueError):
    """An audio descriptor could not be made or read; ``code`` says why.

    The code comes from the protocol's closed vocabulary, so the daemon's
    refusal arms can put it on the wire as it is.
    """

    def __init__(self, message: str, code: str = protocol.ERR_UNAVAILABLE) -> None:
        if code not in protocol.ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}. Use one of: "
                             f"{', '.join(protocol.ERROR_CODES)}.")
        super().__init__(message)
        self.code = code


def _memfd_create(name: str) -> int:
    """Return a new close-on-exec, sealable memfd."""
    flags = _MFD_CLOEXEC | _MFD_ALLOW_SEALING
    create = getattr(os, "memfd_create", None)
    if create is not None:
        return create(name, flags)
    # The Linux UAPI works even when the Python build omits the wrapper.
    try:
        libc_create = ctypes.CDLL(None, use_errno=True).memfd_create
    except (AttributeError, OSError) as error:
        raise OSError(errno.ENOSYS, "memfd_create is unavailable") from error
    libc_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    libc_create.restype = ctypes.c_int
    fd = libc_create(name.encode("ascii"), flags)
    if fd < 0:
        raise OSError(ctypes.get_errno() or errno.ENOSYS, "memfd_create failed")
    return fd


def sealed_readonly(data: bytes) -> int:
    """Return a read-only, close-on-exec descriptor of an immutable copy of ``data``.

    The copy is sealed against write, grow and shrink, and against further
    sealing, before the read-only descriptor is opened, so neither this
    process nor any receiver can change the bytes a descriptor was sent for.
    The caller owns the returned descriptor and must close it. On any failure
    every descriptor made here is closed and DescriptorError (unavailable) is
    raised.
    """
    writer = -1
    try:
        writer = _memfd_create("kilix-voice-audio")
        # Exported views are released before this returns or raises, so a
        # caller that later resizes its buffer is not refused with BufferError.
        with memoryview(data) as view:
            offset = 0
            while offset < len(view):
                with view[offset:] as rest:
                    written = os.write(writer, rest)
                if written <= 0:
                    raise OSError(errno.EIO, "the audio copy stopped making progress")
                offset += written
        fcntl.fcntl(writer, F_ADD_SEALS, ALL_SEALS)
        # Opened only after the seals: the read-only descriptor is the only one
        # that leaves this function.
        return os.open(f"/proc/self/fd/{writer}", os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise DescriptorError(
            f"cannot prepare a sealed audio descriptor: {error}. The system is "
            "out of memory or descriptors; try again after closing some "
            "applications.", protocol.ERR_UNAVAILABLE) from error
    finally:
        if writer >= 0:
            os.close(writer)


def read_descriptor(fd: int, *, byte_limit: int, deadline: float | None = None,
                    clock=time.monotonic) -> bytes:
    """Return the bytes behind a caller's audio descriptor, bounded, or refuse.

    A08: audio arrives as a pre-opened descriptor, never as a path, so the
    daemon reads with the caller's own access and opens nothing on its
    behalf. The read is bounded in bytes and in time and refuses rather than
    truncates. A regular file, a memfd included, is refused unread when it is
    larger than ``byte_limit``. A pipe is read until end-of-file, within
    INGEST_READ_TIMEOUT_S or the caller's ``deadline`` (an absolute instant
    on ``clock``), whichever comes first. A device, a socket, a directory or
    an O_PATH descriptor is refused without being read.

    Whoever calls this -- the daemon's single accept loop -- is never held
    past that bound by a descriptor the caller controls. Every step that
    touches the descriptor, fstat included, runs on a helper thread against a
    duplicate: a descriptor on a filesystem the caller serves can hold even
    fstat or pread indefinitely. This thread waits at most
    INGEST_READ_TIMEOUT_S, or until the deadline if that comes first, plus
    _READ_GRACE_S. A read that has not finished by then is refused and left to
    end on its own, and no more than MAX_STALLED_READS are ever left that way.

    Raises DescriptorError: malformed, unsupported, too-large, deadline,
    unavailable or busy. The caller keeps ownership of ``fd``.
    """
    global _stalled
    with _stalled_lock:
        if _stalled >= MAX_STALLED_READS:
            raise DescriptorError(
                f"{_stalled} earlier audio descriptors are still being read and "
                "have not answered, so this one was not inspected. Attach a "
                "pipe, a memfd or a regular file on a local filesystem, and try "
                "again.", protocol.ERR_BUSY)
    try:
        own = os.dup(fd)
    except OSError as error:
        raise DescriptorError(
            f"cannot inspect the audio descriptor: {error}. Attach an open "
            "descriptor.", protocol.ERR_MALFORMED) from error
    started = clock()
    budget = None if deadline is None else deadline - started
    wait = (INGEST_READ_TIMEOUT_S if budget is None
            else max(0.0, min(INGEST_READ_TIMEOUT_S, budget)))
    result: dict = {}
    state = {"done": False, "abandoned": False}

    def work() -> None:
        global _stalled
        try:
            result["data"] = _read_now(own, byte_limit=byte_limit,
                                       deadline=deadline, clock=clock)
        except BaseException as error:          # handed to the waiting thread
            result["error"] = error
        finally:
            os.close(own)
            with _stalled_lock:
                state["done"] = True
                if state["abandoned"]:
                    _stalled -= 1

    reader = threading.Thread(target=work, name="kilix-voice-ingest-read",
                              daemon=True)
    try:
        reader.start()
    except RuntimeError as error:
        os.close(own)
        raise DescriptorError(
            f"cannot start a thread to read the audio descriptor: {error}. The "
            "system is out of threads; try again later.",
            protocol.ERR_UNAVAILABLE) from error
    reader.join(wait + _READ_GRACE_S)
    with _stalled_lock:
        stalled = not state["done"]
        if stalled:
            state["abandoned"] = True
            _stalled += 1
    if stalled:
        if budget is not None and budget <= INGEST_READ_TIMEOUT_S:
            raise DescriptorError(
                "the request deadline elapsed while the audio descriptor was "
                "being read; nothing was returned.", protocol.ERR_DEADLINE)
        raise DescriptorError(
            f"the audio descriptor did not answer within "
            f"{INGEST_READ_TIMEOUT_S:g} s and was refused. Attach a pipe, a "
            "memfd or a regular file on a local filesystem.",
            protocol.ERR_UNAVAILABLE)
    if "error" in result:
        raise result["error"]
    return result["data"]


def _read_now(fd: int, *, byte_limit: int, deadline: float | None,
              clock) -> bytes:
    """The read itself, run on read_descriptor's helper thread."""
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        info = os.fstat(fd)
    except OSError as error:
        raise DescriptorError(
            f"cannot inspect the audio descriptor: {error}. Attach an open "
            "descriptor.", protocol.ERR_MALFORMED) from error
    # Checked before anything reads or reopens the descriptor. An O_PATH
    # descriptor's access mode reads as O_RDONLY, yet it grants no read, and
    # reopening either kind through /proc would read what the caller never
    # opened for reading.
    if flags & _O_PATH:
        raise DescriptorError(
            "the audio descriptor was opened with O_PATH, which grants no read "
            "access. Attach a descriptor opened for reading.",
            protocol.ERR_MALFORMED)
    if (flags & os.O_ACCMODE) not in (os.O_RDONLY, os.O_RDWR):
        raise DescriptorError(
            "the audio descriptor is not open for reading. Attach a read-only "
            "descriptor, or the read end of the pipe.", protocol.ERR_MALFORMED)
    if stat.S_ISREG(info.st_mode):
        return _read_file(fd, info.st_size, byte_limit)
    if stat.S_ISFIFO(info.st_mode):
        return _read_pipe(fd, byte_limit, deadline, clock)
    raise DescriptorError(
        "the audio descriptor must be a pipe, a regular file or a memfd. A "
        "device, socket or directory is refused without being read.",
        protocol.ERR_UNSUPPORTED)


def _unreadable(error: OSError) -> DescriptorError:
    """Code a read the descriptor refused: the caller's descriptor or the system."""
    reason = os.strerror(error.errno) if error.errno else str(error)
    if error.errno in (errno.EBADF, errno.EINVAL, errno.EISDIR, errno.ESPIPE):
        return DescriptorError(
            f"the audio descriptor cannot be read ({reason}). Attach a pipe, a "
            "regular file or a memfd opened for reading.", protocol.ERR_MALFORMED)
    return DescriptorError(
        f"reading the audio descriptor failed ({reason}); nothing was returned. "
        "Try again, or attach the audio as a memfd.", protocol.ERR_UNAVAILABLE)


def _read_file(fd: int, size: int, byte_limit: int) -> bytes:
    if size > byte_limit:
        raise DescriptorError(
            f"the audio is {size} bytes and the byte limit is "
            f"{byte_limit}. It was refused without being read.",
            protocol.ERR_TOO_LARGE)
    chunks, total = [], 0
    try:
        while total <= byte_limit:
            block = os.pread(fd, min(_READ_BLOCK, byte_limit + 1 - total), total)
            if not block:
                break
            chunks.append(block)
            total += len(block)
    except OSError as error:
        raise _unreadable(error) from error
    if total > byte_limit:
        raise DescriptorError(
            f"the audio grew past the byte limit of {byte_limit} while it "
            "was read. It was refused, not truncated.", protocol.ERR_TOO_LARGE)
    return b"".join(chunks)


def _read_pipe(fd: int, byte_limit: int, deadline: float | None, clock) -> bytes:
    # The descriptor's open file description is the SENDER's too. Read
    # blocking, it stalled whenever the sender took the bytes a readiness check
    # had just seen. Setting O_NONBLOCK on it would change the sender's own end,
    # and a sender that had set it made the read raise EAGAIN. A private
    # description of the same pipe, opened non-blocking, has neither problem.
    try:
        mine = os.open(f"/proc/self/fd/{fd}",
                       os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as error:
        raise DescriptorError(
            f"cannot open a private read end of the audio pipe "
            f"({os.strerror(error.errno) if error.errno else error}). Attach a "
            "regular file or a memfd instead.", protocol.ERR_UNSUPPORTED) from error
    try:
        poller = select.poll()
        poller.register(mine, select.POLLIN)
        started = clock()
        chunks, total = [], 0
        while True:
            now = clock()
            window = started + INGEST_READ_TIMEOUT_S - now
            budget = None if deadline is None else deadline - now
            left = window if budget is None else min(window, budget)
            if left <= 0:
                if budget is not None and budget <= window:
                    raise DescriptorError(
                        "the request deadline elapsed while the audio pipe was "
                        "being read; nothing was returned.", protocol.ERR_DEADLINE)
                raise DescriptorError(
                    f"the audio pipe did not reach end-of-file within "
                    f"{INGEST_READ_TIMEOUT_S:g} s. Close the write end once the "
                    "audio is written, or attach a regular file or memfd.",
                    protocol.ERR_MALFORMED)
            if not poller.poll(int(left * 1000) + 1):
                continue
            try:
                block = os.read(mine, min(_PIPE_BLOCK, byte_limit + 1 - total))
            except BlockingIOError:
                # The sender read what poll saw first: not an error, and no
                # reason to wait past the bound either.
                continue
            except OSError as error:
                raise _unreadable(error) from error
            if not block:
                return b"".join(chunks)
            chunks.append(block)
            total += len(block)
            if total > byte_limit:
                # Stop here: never drain the rest of a pipe that is too big.
                raise DescriptorError(
                    f"the audio pipe carried more than the byte limit of "
                    f"{byte_limit}. It was refused, not truncated.",
                    protocol.ERR_TOO_LARGE)
    finally:
        os.close(mine)
