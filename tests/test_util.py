"""Offline tests for voicelib.util and voicelib.paths.

Nothing here reads or writes the developer's real Kilix tree: the environment
is replaced wholesale and every path lives under a temporary directory.  No
audio device is opened and no clip is played — WAV streams are assembled byte
by byte so the parser is exercised against known headers rather than against
whatever an engine happens to emit on this machine.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import struct
import tempfile
import unittest
from unittest import mock

from voicelib import paths, util

_FMT_PCM = 1
_FMT_IEEE_FLOAT = 3


def build_wav(pcm: bytes, rate: int, *, channels: int = 1, bits: int = 16,
              tag: int = _FMT_PCM, data_size: int | None = None,
              before_data: bytes = b"", after_data: bytes = b"") -> bytes:
    """Assemble a RIFF/WAVE stream, including deliberately malformed ones."""
    block_align = channels * bits // 8
    fmt_body = struct.pack("<HHIIHH", tag, channels, rate,
                           rate * block_align, block_align, bits)
    body = (b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
            + before_data
            + b"data" + struct.pack("<I", len(pcm) if data_size is None
                                    else data_size) + pcm
            + after_data)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def tone(sample: int, count: int) -> bytes:
    """Return ``count`` s16le samples of one constant value."""
    return struct.pack("<h", sample) * count


class CfgGetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = {"audio": {"rate": 16000, "frame_ms": 20, "capture_cmd": None},
                    "tts": {"engine": {"name": "espeak"}},
                    "top": 1}

    def test_returns_nested_value(self) -> None:
        self.assertEqual(util.cfg_get(self.cfg, "audio.rate"), 16000)
        self.assertEqual(util.cfg_get(self.cfg, "tts.engine.name"), "espeak")

    def test_returns_top_level_value(self) -> None:
        self.assertEqual(util.cfg_get(self.cfg, "top"), 1)

    def test_stored_none_is_returned_not_defaulted(self) -> None:
        self.assertIsNone(util.cfg_get(self.cfg, "audio.capture_cmd", ["parec"]))

    def test_missing_key_returns_default(self) -> None:
        self.assertEqual(util.cfg_get(self.cfg, "audio.missing", 7), 7)
        self.assertEqual(util.cfg_get(self.cfg, "missing.rate", 7), 7)
        self.assertIsNone(util.cfg_get(self.cfg, "nope"))

    def test_descending_through_a_non_dict_returns_default(self) -> None:
        self.assertEqual(util.cfg_get(self.cfg, "top.deeper", "d"), "d")

    def test_empty_config_returns_default(self) -> None:
        self.assertEqual(util.cfg_get({}, "audio.rate", 16000), 16000)


class Rms16TestCase(unittest.TestCase):

    def test_empty_frame_is_zero(self) -> None:
        self.assertEqual(util.rms16(b""), 0.0)

    def test_single_byte_is_zero(self) -> None:
        # A truncated final read leaves half a sample; that is normal, not an
        # error, and there is no whole sample to measure.
        self.assertEqual(util.rms16(b"\x7f"), 0.0)

    def test_silence_is_zero(self) -> None:
        self.assertEqual(util.rms16(tone(0, 320)), 0.0)

    def test_known_level(self) -> None:
        self.assertAlmostEqual(util.rms16(tone(16384, 320)), 0.5, places=6)
        self.assertAlmostEqual(util.rms16(tone(-16384, 320)), 0.5, places=6)
        self.assertAlmostEqual(util.rms16(tone(8192, 10)), 0.25, places=6)

    def test_full_scale_is_one(self) -> None:
        self.assertAlmostEqual(util.rms16(tone(-32768, 16)), 1.0, places=6)

    def test_alternating_signs_measure_amplitude_not_mean(self) -> None:
        frame = (struct.pack("<h", 8192) + struct.pack("<h", -8192)) * 160
        self.assertAlmostEqual(util.rms16(frame), 0.25, places=6)

    def test_odd_trailing_byte_is_ignored(self) -> None:
        self.assertAlmostEqual(util.rms16(tone(16384, 4) + b"\x7f"), 0.5,
                               places=6)

    def test_louder_frame_measures_higher(self) -> None:
        self.assertGreater(util.rms16(tone(20000, 32)), util.rms16(tone(200, 32)))

    def test_reads_little_endian_regardless_of_host(self) -> None:
        # b"\x00\x40" is 16384 little-endian and 64 big-endian; the frozen wire
        # format is s16le, so the answer must not depend on the machine.
        self.assertAlmostEqual(util.rms16(b"\x00\x40" * 8), 0.5, places=6)


class WriteWavTestCase(unittest.TestCase):

    def test_header_describes_mono_s16le(self) -> None:
        blob = util.write_wav(tone(1000, 100), 16000)
        self.assertEqual(blob[:4], b"RIFF")
        self.assertEqual(blob[8:12], b"WAVE")
        tag, channels, rate, _byte_rate, _align, bits = struct.unpack_from(
            "<HHIIHH", blob, 20)
        self.assertEqual((tag, channels, rate, bits), (_FMT_PCM, 1, 16000, 16))

    def test_odd_pcm_tail_is_dropped(self) -> None:
        pcm, rate = util.parse_wav_bytes(util.write_wav(b"\x01\x02\x03", 16000))
        self.assertEqual(pcm, b"\x01\x02")
        self.assertEqual(rate, 16000)

    def test_empty_pcm_is_a_valid_empty_clip(self) -> None:
        pcm, rate = util.parse_wav_bytes(util.write_wav(b"", 22050))
        self.assertEqual(pcm, b"")
        self.assertEqual(rate, 22050)

    def test_non_positive_rate_is_refused(self) -> None:
        for rate in (0, -1):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError) as caught:
                    util.write_wav(tone(1, 8), rate)
                self.assertIn("rate", str(caught.exception))


class ParseWavTestCase(unittest.TestCase):

    def test_round_trip_through_write_wav(self) -> None:
        for rate in (8000, 16000, 22050, 44100):
            with self.subTest(rate=rate):
                original = tone(4321, 500)
                pcm, parsed_rate = util.parse_wav_bytes(
                    util.write_wav(original, rate))
                self.assertEqual(pcm, original)
                self.assertEqual(parsed_rate, rate)

    def test_optional_chunks_before_data_are_skipped(self) -> None:
        pcm = tone(999, 64)
        extra = (b"fact" + struct.pack("<I", 4) + struct.pack("<I", 64)
                 + b"LIST" + struct.pack("<I", 4) + b"INFO")
        parsed, rate = util.parse_wav_bytes(
            build_wav(pcm, 16000, before_data=extra))
        self.assertEqual(parsed, pcm)
        self.assertEqual(rate, 16000)

    def test_odd_sized_chunk_pad_byte_is_skipped(self) -> None:
        pcm = tone(5, 16)
        extra = b"note" + struct.pack("<I", 3) + b"abc" + b"\x00"
        parsed, _rate = util.parse_wav_bytes(
            build_wav(pcm, 16000, before_data=extra))
        self.assertEqual(parsed, pcm)

    def test_declared_length_bounds_the_payload(self) -> None:
        pcm = tone(7, 32)
        blob = build_wav(pcm, 16000, after_data=b"LIST" + struct.pack("<I", 0))
        parsed, _rate = util.parse_wav_bytes(blob)
        self.assertEqual(parsed, pcm)

    def test_placeholder_length_falls_back_to_bytes_received(self) -> None:
        # espeak-ng piping to stdout cannot seek back to patch its header.
        pcm = tone(11, 40)
        for declared in (0, 0x7FFFFF80, 0xFFFFFFFF):
            with self.subTest(declared=declared):
                parsed, _rate = util.parse_wav_bytes(
                    build_wav(pcm, 22050, data_size=declared))
                self.assertEqual(parsed, pcm)

    def test_truncated_payload_returns_what_arrived(self) -> None:
        pcm = tone(3, 100)
        blob = build_wav(pcm, 16000)
        parsed, rate = util.parse_wav_bytes(blob[:len(blob) - 60])
        self.assertEqual(rate, 16000)
        self.assertEqual(parsed, pcm[:len(pcm) - 60])

    def test_truncated_header_is_rejected(self) -> None:
        blob = util.write_wav(tone(3, 100), 16000)
        with self.assertRaises(ValueError) as caught:
            util.parse_wav_bytes(blob[:20])
        self.assertIn("fmt", str(caught.exception))

    def test_truncated_before_any_chunk_is_rejected(self) -> None:
        blob = util.write_wav(tone(3, 100), 16000)
        for cut in (0, 4, 11):
            with self.subTest(cut=cut):
                with self.assertRaises(ValueError):
                    util.parse_wav_bytes(blob[:cut])

    def test_non_riff_blob_is_rejected(self) -> None:
        for blob in (b"", b"not a wav at all", b"\x00" * 64,
                     b"RIFF" + struct.pack("<I", 4) + b"AVI ",
                     b"OggS" + b"\x00" * 40):
            with self.subTest(blob=blob[:8]):
                with self.assertRaises(ValueError) as caught:
                    util.parse_wav_bytes(blob)
                self.assertTrue(str(caught.exception))

    def test_missing_data_chunk_is_rejected(self) -> None:
        fmt_body = struct.pack("<HHIIHH", _FMT_PCM, 1, 16000, 32000, 2, 16)
        body = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
        blob = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
        with self.assertRaises(ValueError) as caught:
            util.parse_wav_bytes(blob)
        self.assertIn("data", str(caught.exception))

    def test_data_before_fmt_is_rejected(self) -> None:
        pcm = tone(1, 8)
        fmt_body = struct.pack("<HHIIHH", _FMT_PCM, 1, 16000, 32000, 2, 16)
        body = (b"data" + struct.pack("<I", len(pcm)) + pcm
                + b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body)
        blob = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
        with self.assertRaises(ValueError):
            util.parse_wav_bytes(blob)

    def test_unsupported_shapes_are_rejected(self) -> None:
        pcm = tone(1, 32)
        cases = {
            "stereo": dict(channels=2),
            "eight_bit": dict(bits=8),
            "float": dict(tag=_FMT_IEEE_FLOAT),
            "zero_rate": dict(rate=0),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                kwargs.setdefault("rate", 16000)
                rate = kwargs.pop("rate")
                with self.assertRaises(ValueError) as caught:
                    util.parse_wav_bytes(build_wav(pcm, rate, **kwargs))
                self.assertTrue(str(caught.exception))


class WhichTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="kilix-voice-which-")
        self.addCleanup(_rmtree, self.dir)
        patcher = mock.patch.dict(os.environ, {"PATH": self.dir})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _make(self, name: str, mode: int) -> str:
        target = os.path.join(self.dir, name)
        with open(target, "w", encoding="utf-8") as stream:
            stream.write("#!/bin/sh\nexit 0\n")
        os.chmod(target, mode)
        return target

    def test_finds_an_executable_on_path(self) -> None:
        target = self._make("kilix-voice-fake-engine", 0o755)
        self.assertEqual(util.which("kilix-voice-fake-engine"), target)

    def test_non_executable_file_is_not_a_command(self) -> None:
        self._make("kilix-voice-not-runnable", 0o644)
        self.assertIsNone(util.which("kilix-voice-not-runnable"))

    def test_missing_command_is_none(self) -> None:
        self.assertIsNone(util.which("kilix-voice-absolutely-not-installed"))


class RepoRootTestCase(unittest.TestCase):

    def test_points_at_the_checkout(self) -> None:
        root = util.repo_root()
        self.assertTrue(root.is_absolute())
        self.assertTrue((root / "VERSION").is_file())
        self.assertTrue((root / "Makefile").is_file())
        self.assertTrue((root / "voicelib" / "util.py").is_file())

    def test_is_a_path_object(self) -> None:
        self.assertIsInstance(util.repo_root(), pathlib.Path)


class PathsTestCase(unittest.TestCase):
    """Path resolution against a synthetic environment only."""

    def setUp(self) -> None:
        self.home = os.path.realpath(
            tempfile.mkdtemp(prefix="kilix-voice-home-"))
        self.addCleanup(_rmtree, self.home)

    def use_env(self, **overrides: str) -> None:
        env = {"HOME": self.home}
        env.update(overrides)
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults_match_the_documented_layout(self) -> None:
        self.use_env()
        base = os.path.join(self.home, ".local", "gpu_terminal")
        self.assertEqual(paths.gpu_terminal_home(), base)
        self.assertEqual(paths.settings_file(),
                         os.path.join(base, "settings.conf"))
        self.assertEqual(paths.session_dir(),
                         os.path.join(base, "kilix", "session", "voice"))
        self.assertEqual(paths.data_dir(),
                         os.path.join(base, "kilix", "data", "voice"))

    def test_settings_override_wins_over_home(self) -> None:
        self.use_env(GPU_TERMINAL_HOME=os.path.join(self.home, "gt"),
                     GPU_TERMINAL_SETTINGS_FILE=os.path.join(self.home, "s.conf"))
        self.assertEqual(paths.settings_file(),
                         os.path.join(self.home, "s.conf"))

    def test_settings_follows_gpu_terminal_home(self) -> None:
        self.use_env(GPU_TERMINAL_HOME=os.path.join(self.home, "gt"))
        self.assertEqual(paths.settings_file(),
                         os.path.join(self.home, "gt", "settings.conf"))

    def test_session_and_data_overrides(self) -> None:
        session = os.path.join(self.home, "run", "sess")
        data = os.path.join(self.home, "share", "data")
        self.use_env(KILIX_SESSION_HOME=session, KILIX_DATA_HOME=data)
        self.assertEqual(paths.session_dir(), os.path.join(session, "voice"))
        self.assertEqual(paths.data_dir(), os.path.join(data, "voice"))
        self.assertEqual(paths.control_socket(),
                         os.path.join(session, "voice", "control.sock"))
        self.assertEqual(paths.libvosk_path(),
                         os.path.join(data, "voice", "lib", "current",
                                      "libvosk.so"))
        self.assertEqual(paths.models_dir(),
                         os.path.join(data, "voice", "models"))
        self.assertEqual(paths.model_dir("small-en-us"),
                         os.path.join(data, "voice", "models", "small-en-us"))

    def test_storage_home_relocates_both_trees(self) -> None:
        storage = os.path.join(self.home, "elsewhere")
        self.use_env(KILIX_STORAGE_HOME=storage)
        self.assertEqual(paths.session_dir(),
                         os.path.join(storage, "session", "voice"))
        self.assertEqual(paths.data_dir(),
                         os.path.join(storage, "data", "voice"))

    def test_tilde_and_relative_values_are_expanded(self) -> None:
        self.use_env(KILIX_SESSION_HOME="~/tilde-session")
        self.assertEqual(paths.session_dir(),
                         os.path.join(self.home, "tilde-session", "voice"))

    def test_dictate_socket_names_a_leaf_in_the_session_dir(self) -> None:
        self.use_env(KILIX_SESSION_HOME=os.path.join(self.home, "sess"))
        session = paths.session_dir()
        for pane in ("3", 3, "pane_1", "A-b_9", "x" * 64):
            with self.subTest(pane=pane):
                target = paths.dictate_socket(pane)
                self.assertEqual(os.path.dirname(target), session)
                self.assertEqual(os.path.basename(target),
                                 f"dictate-{pane}.sock")

    def test_dictate_socket_refuses_a_pane_id_that_is_a_path(self) -> None:
        self.use_env()
        for pane in ("", "..", ".", "a/b", "../../etc/passwd", "pane id",
                     "x" * 65, "a.b", "a\x00b"):
            with self.subTest(pane=pane):
                with self.assertRaises(paths.PathError) as caught:
                    paths.dictate_socket(pane)
                self.assertIn("pane id", str(caught.exception))

    def test_model_dir_refuses_an_id_that_is_a_path(self) -> None:
        self.use_env()
        for model in ("", "..", ".hidden", "a/b", "../../etc", "m odel",
                      "x" * 65, "-leading"):
            with self.subTest(model=model):
                with self.assertRaises(paths.PathError) as caught:
                    paths.model_dir(model)
                self.assertIn("model id", str(caught.exception))

    def test_model_dir_accepts_catalog_ids(self) -> None:
        self.use_env()
        for model in ("small-en-us", "lgraph-en-us", "vibevoice-asr-bitnet",
                      "vosk-model-small-en-us-0.15"):
            with self.subTest(model=model):
                self.assertEqual(
                    os.path.dirname(paths.model_dir(model)), paths.models_dir())


class EnsurePrivateDirTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.root = os.path.realpath(
            tempfile.mkdtemp(prefix="kilix-voice-dirs-"))
        self.addCleanup(_rmtree, self.root)

    def _mode(self, path: str) -> int:
        return stat.S_IMODE(os.stat(path).st_mode)

    def test_creates_a_private_directory_despite_a_loose_umask(self) -> None:
        previous = os.umask(0o000)
        self.addCleanup(os.umask, previous)
        target = paths.ensure_private_dir(os.path.join(self.root, "voice"))
        self.assertTrue(os.path.isdir(target))
        self.assertEqual(self._mode(target), 0o700)

    def test_creates_missing_parents(self) -> None:
        target = paths.ensure_private_dir(
            os.path.join(self.root, "a", "b", "voice"))
        self.assertTrue(os.path.isdir(target))
        self.assertEqual(self._mode(target), 0o700)

    def test_is_idempotent(self) -> None:
        first = paths.ensure_private_dir(os.path.join(self.root, "voice"))
        second = paths.ensure_private_dir(os.path.join(self.root, "voice"))
        self.assertEqual(first, second)
        self.assertEqual(self._mode(second), 0o700)

    def test_honours_an_explicit_mode(self) -> None:
        target = os.path.join(self.root, "custom")
        paths.ensure_private_dir(target, 0o750)
        self.assertEqual(self._mode(target), 0o750)
        paths.ensure_private_dir(target, 0o750)

    def test_world_readable_directory_is_reported_not_repaired(self) -> None:
        target = os.path.join(self.root, "loose")
        os.mkdir(target)
        os.chmod(target, 0o755)  # explicit: the process umask would trim mkdir's mode
        with self.assertRaises(paths.PathError) as caught:
            paths.ensure_private_dir(target)
        message = str(caught.exception)
        self.assertIn("0755", message)
        self.assertIn("chmod", message)
        self.assertEqual(self._mode(target), 0o755)

    def test_symlink_in_the_final_position_is_refused(self) -> None:
        real = os.path.join(self.root, "real")
        os.mkdir(real, 0o700)
        link = os.path.join(self.root, "voice")
        os.symlink(real, link)
        with self.assertRaises(paths.PathError) as caught:
            paths.ensure_private_dir(link)
        self.assertIn("symlink", str(caught.exception))

    def test_regular_file_in_the_way_is_refused(self) -> None:
        target = os.path.join(self.root, "voice")
        with open(target, "w", encoding="utf-8") as stream:
            stream.write("not a directory\n")
        with self.assertRaises(paths.PathError):
            paths.ensure_private_dir(target)

    def test_unwritable_parent_is_reported(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        parent = os.path.join(self.root, "readonly")
        os.mkdir(parent, 0o500)
        self.addCleanup(os.chmod, parent, 0o700)
        with self.assertRaises(paths.PathError) as caught:
            paths.ensure_private_dir(os.path.join(parent, "voice"))
        self.assertIn("cannot create", str(caught.exception))


def _rmtree(path: str) -> None:
    """Remove a temporary tree, restoring modes the tests deliberately broke."""
    for base, directories, _files in os.walk(path):
        for name in directories:
            try:
                os.chmod(os.path.join(base, name), 0o700)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
