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
    on ``clock``), whichever comes first. A device, a socket or a directory
    is refused without being read.

    Raises DescriptorError: malformed, unsupported, too-large or deadline.
    The caller keeps ownership of ``fd``.
    """
    try:
        info = os.fstat(fd)
        access = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
    except OSError as error:
        raise DescriptorError(
            f"cannot inspect the audio descriptor: {error}. Attach an open "
            "descriptor.", protocol.ERR_MALFORMED) from error
    if access not in (os.O_RDONLY, os.O_RDWR):
        raise DescriptorError(
            "the audio descriptor is not open for reading. Attach a read-only "
            "descriptor, or the read end of the pipe.", protocol.ERR_MALFORMED)
    if stat.S_ISREG(info.st_mode):
        if info.st_size > byte_limit:
            raise DescriptorError(
                f"the audio is {info.st_size} bytes and the byte limit is "
                f"{byte_limit}. It was refused without being read.",
                protocol.ERR_TOO_LARGE)
        chunks, total = [], 0
        while total <= byte_limit:
            block = os.pread(fd, min(_READ_BLOCK, byte_limit + 1 - total), total)
            if not block:
                break
            chunks.append(block)
            total += len(block)
        if total > byte_limit:
            raise DescriptorError(
                f"the audio grew past the byte limit of {byte_limit} while it "
                "was read. It was refused, not truncated.", protocol.ERR_TOO_LARGE)
        return b"".join(chunks)
    if stat.S_ISFIFO(info.st_mode):
        # Never O_NONBLOCK: the descriptor shares its open file description
        # with the sender, whose own end would change under it.
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
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                continue
            block = os.read(fd, min(_PIPE_BLOCK, byte_limit + 1 - total))
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
    raise DescriptorError(
        "the audio descriptor must be a pipe, a regular file or a memfd. A "
        "device, socket or directory is refused without being read.",
        protocol.ERR_UNSUPPORTED)
