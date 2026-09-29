"""The consent payload digest cache follows git's racy-index rule.

VibeVoice's payload is 1.7 GB, so the per-file digest is remembered -- but only
for a file already RACY_SECONDS old when hashed. A just-written file is never
cached, which is what keeps two equal-length writes in one timestamp tick from
sharing a stale digest (see PayloadBindingTestCase in test_p1_mechanisms).
"""
import os
import tempfile
import time
import unittest
from unittest import mock

from voicelib import consent


class DigestCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "weights.gguf")
        with open(self.path, "wb") as handle:
            handle.write(b"first payload")
        consent._DIGESTS.clear()
        self.addCleanup(consent._DIGESTS.clear)

    def later(self):
        # As if the clock had moved on, so the file counts as settled.
        now = time.time_ns() + 10 * 1_000_000_000
        return mock.patch.object(consent.time, "time_ns", return_value=now)

    def test_a_file_written_moments_ago_is_never_cached(self) -> None:
        consent._file_digest(self.path)
        self.assertEqual(consent._DIGESTS, {})

    def test_a_settled_file_is_hashed_once(self) -> None:
        with self.later():
            first = consent._file_digest(self.path)
            self.assertEqual(len(consent._DIGESTS), 1)
            with mock.patch.object(consent.hashlib, "sha256",
                                   side_effect=AssertionError("re-hashed")):
                self.assertEqual(consent._file_digest(self.path), first)

    def test_any_rewrite_misses_the_cache(self) -> None:
        with self.later():
            first = consent._file_digest(self.path)
        # The rule's premise: a cached file was RACY_SECONDS old, so any later
        # write lands in a later timestamp tick. Faking the clock above skips
        # that wait, so let real time pass before the write to restore it.
        time.sleep(0.05)
        with open(self.path, "wb") as handle:
            handle.write(b"other payload")      # equal length, new ctime
        self.assertNotEqual(consent._file_digest(self.path), first)

    def test_a_replaced_file_misses_the_cache(self) -> None:
        with self.later():
            first = consent._file_digest(self.path)
        replacement = self.path + ".new"
        with open(replacement, "wb") as handle:
            handle.write(b"third payload")
        os.replace(replacement, self.path)      # new inode
        self.assertNotEqual(consent._file_digest(self.path), first)


if __name__ == "__main__":
    unittest.main()
