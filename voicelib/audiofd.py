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

from . import protocol

# At most this many SCM_RIGHTS descriptors are received with one control
# request. More are closed by the kernel as the message is truncated.
MAX_INBOUND_FDS = 4

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
