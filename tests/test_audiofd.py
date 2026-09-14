"""AUD-01: the sealed, read-only audio descriptor.

Every property is checked on a real memfd: the bytes read back, a write
refused, all four seals present, close-on-exec set, and no descriptor left
open when making one fails.
"""

from __future__ import annotations

import errno
import fcntl
import os
import unittest
from unittest import mock

from voicelib import audiofd, protocol


def _open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


class SealedReadonly(unittest.TestCase):

    def make(self, data: bytes) -> int:
        fd = audiofd.sealed_readonly(data)
        self.addCleanup(os.close, fd)
        return fd

    def assert_sealed_readonly(self, fd: int, data: bytes) -> None:
        self.assertEqual(os.pread(fd, len(data) + 1, 0), data)
        with self.assertRaises(OSError):
            os.write(fd, b"x")                           # a read-only descriptor
        # Reopening the file for writing is allowed; the seals then refuse
        # every change through that descriptor too: no write, no resize.
        writable = os.open(f"/proc/self/fd/{fd}", os.O_WRONLY)
        try:
            with self.assertRaises(PermissionError):
                os.write(writable, b"x")
            with self.assertRaises(PermissionError):
                os.ftruncate(writable, len(data) + 4096)
            if data:
                with self.assertRaises(PermissionError):
                    os.ftruncate(writable, 0)
        finally:
            os.close(writable)
        self.assertEqual(fcntl.fcntl(fd, audiofd.F_GET_SEALS) & audiofd.ALL_SEALS,
                         audiofd.ALL_SEALS)
        self.assertEqual(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE, os.O_RDONLY)
        self.assertFalse(os.get_inheritable(fd))                # close-on-exec

    def test_bytes_identical_write_refused_all_four_seals(self) -> None:
        data = bytes(range(256)) * 400                           # > one page
        self.assert_sealed_readonly(self.make(data), data)

    def test_an_empty_clip_is_still_a_sealed_descriptor(self) -> None:
        self.assert_sealed_readonly(self.make(b""), b"")

    def test_failure_after_create_closes_both(self) -> None:
        before = _open_descriptors()
        with mock.patch.object(audiofd.fcntl, "fcntl",
                               side_effect=OSError(errno.EPERM, "no seals")):
            with self.assertRaises(audiofd.DescriptorError) as caught:
                audiofd.sealed_readonly(b"abc")
        self.assertEqual(caught.exception.code, protocol.ERR_UNAVAILABLE)
        self.assertEqual(_open_descriptors(), before)

    def test_failure_opening_the_reader_closes_the_writer(self) -> None:
        before = _open_descriptors()
        with mock.patch.object(audiofd.os, "open",
                               side_effect=OSError(errno.EMFILE, "no descriptors")):
            with self.assertRaises(audiofd.DescriptorError):
                audiofd.sealed_readonly(b"abc")
        self.assertEqual(_open_descriptors(), before)

    def test_ctypes_fallback_when_os_memfd_create_absent(self) -> None:
        saved = os.memfd_create
        del os.memfd_create
        try:
            self.assertFalse(hasattr(os, "memfd_create"))
            fd = self.make(b"through libc")
        finally:
            os.memfd_create = saved
        self.assert_sealed_readonly(fd, b"through libc")

    def test_a_code_outside_the_vocabulary_cannot_be_attached(self) -> None:
        with self.assertRaises(ValueError):
            audiofd.DescriptorError("x", "oops")
        self.assertEqual(audiofd.DescriptorError("x", protocol.ERR_TOO_LARGE).code,
                         protocol.ERR_TOO_LARGE)


if __name__ == "__main__":
    unittest.main()
