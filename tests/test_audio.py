"""Offline tests for voicelib.audio: command building and the no-device rule.

``build_capture_cmd`` and ``build_play_cmd`` are pure functions of the config
and of PATH, so PATH is replaced with a fake ``which`` and the argv is asserted
exactly.  Nothing here starts a recorder or a sink: where a launch would
otherwise be possible, ``subprocess.Popen`` is replaced with something that
fails the test if it is ever called.  No microphone is opened and no sound is
produced.
"""

from __future__ import annotations

import unittest
from unittest import mock

from voicelib import audio


def path_with(*names: str):
    """Return a ``which`` that finds only ``names``."""
    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in names else None
    return which


def no_launch() -> mock._patch:
    """Patch Popen so that any attempt to start a process fails the test."""
    return mock.patch.object(
        audio.subprocess, "Popen",
        side_effect=AssertionError("no process may be launched here"))


class CaptureCommand(unittest.TestCase):
    """build_capture_cmd: the recorder argv, chosen from what is installed."""

    def test_parec_is_preferred(self) -> None:
        with mock.patch.object(audio, "which", path_with("parec", "arecord")):
            self.assertEqual(
                audio.build_capture_cmd({}),
                ["parec", "--format=s16le", "--rate=16000", "--channels=1",
                 "--latency-msec=30"])

    def test_parec_takes_the_configured_source(self) -> None:
        cfg = {"audio": {"rate": 48000,
                         "device_in": "alsa_input.pci-0000_00_1f.3.analog"}}
        with mock.patch.object(audio, "which", path_with("parec")):
            self.assertEqual(
                audio.build_capture_cmd(cfg),
                ["parec", "--format=s16le", "--rate=48000", "--channels=1",
                 "--latency-msec=30", "-d",
                 "alsa_input.pci-0000_00_1f.3.analog"])

    def test_the_default_source_adds_no_device_argument(self) -> None:
        with mock.patch.object(audio, "which", path_with("parec")):
            cmd = audio.build_capture_cmd({"audio": {"device_in": "default"}})
        self.assertNotIn("-d", cmd)

    def test_arecord_is_the_fallback(self) -> None:
        # The ALSA path takes no device by contract: KILIX_VOICE_DEVICE_IN
        # names a PulseAudio source, which means nothing to arecord.
        cfg = {"audio": {"rate": 44100, "device_in": "some_source"}}
        with mock.patch.object(audio, "which", path_with("arecord")):
            self.assertEqual(
                audio.build_capture_cmd(cfg),
                ["arecord", "-q", "-f", "S16_LE", "-r", "44100", "-c", "1",
                 "-t", "raw"])

    def test_no_recorder_at_all_is_actionable(self) -> None:
        with mock.patch.object(audio, "which", path_with()):
            with self.assertRaises(audio.AudioError) as caught:
                audio.build_capture_cmd({})
        message = str(caught.exception)
        self.assertIn("parec", message)
        self.assertIn("arecord", message)
        self.assertIn("apt install", message)
        self.assertIn("audio.capture_cmd", message)

    def test_the_default_rate_is_the_frozen_one(self) -> None:
        with mock.patch.object(audio, "which", path_with("parec")):
            self.assertIn(f"--rate={audio.DEFAULT_RATE}",
                          audio.build_capture_cmd(None))


class PlayCommand(unittest.TestCase):
    """build_play_cmd: the sink argv, at the rate of the clip in hand."""

    def test_pacat_is_preferred(self) -> None:
        with mock.patch.object(audio, "which", path_with("pacat", "aplay")):
            self.assertEqual(
                audio.build_play_cmd({}, 22050),
                ["pacat", "--playback", "--format=s16le", "--rate=22050",
                 "--channels=1"])

    def test_pacat_takes_the_configured_sink(self) -> None:
        cfg = {"audio": {"device_out": "alsa_output.usb-1.2.analog-stereo"}}
        with mock.patch.object(audio, "which", path_with("pacat")):
            self.assertEqual(
                audio.build_play_cmd(cfg, 16000)[-2:],
                ["-d", "alsa_output.usb-1.2.analog-stereo"])

    def test_aplay_is_the_fallback(self) -> None:
        with mock.patch.object(audio, "which", path_with("aplay")):
            self.assertEqual(
                audio.build_play_cmd({}, 22050),
                ["aplay", "-q", "-f", "S16_LE", "-r", "22050", "-c", "1",
                 "-t", "raw"])

    def test_the_rate_is_per_clip_not_per_config(self) -> None:
        # espeak-ng synthesises at 22.05 kHz while capture runs at 16 kHz; the
        # sink is told which one this clip is.
        cfg = {"audio": {"rate": 16000}}
        with mock.patch.object(audio, "which", path_with("pacat")):
            self.assertIn("--rate=22050", audio.build_play_cmd(cfg, 22050))

    def test_no_sink_at_all_is_actionable(self) -> None:
        with mock.patch.object(audio, "which", path_with()):
            with self.assertRaises(audio.AudioError) as caught:
                audio.build_play_cmd({}, 22050)
        message = str(caught.exception)
        self.assertIn("pacat", message)
        self.assertIn("aplay", message)
        self.assertIn("audio.play_cmd", message)

    def test_an_unusable_clip_rate_is_refused(self) -> None:
        for bad in (0, -1, "fast", None):
            with self.subTest(rate=bad):
                with self.assertRaises(audio.AudioError) as caught:
                    audio.build_play_cmd({}, bad)
                self.assertIn("clip sample rate", str(caught.exception))


class Overrides(unittest.TestCase):
    """The command overrides are the seam a fake device is injected on."""

    def test_capture_override_substitutes_rate_and_device(self) -> None:
        cfg = {"audio": {"rate": 8000, "device_in": "mic0",
                         "capture_cmd": ["cat", "--rate={rate}", "{device}"]}}
        with mock.patch.object(audio, "which", path_with("parec")):
            self.assertEqual(audio.build_capture_cmd(cfg),
                             ["cat", "--rate=8000", "mic0"])

    def test_play_override_substitutes_rate_and_device(self) -> None:
        cfg = {"audio": {"device_out": "sink0",
                         "play_cmd": ("tee", "{device}-{rate}.raw")}}
        with mock.patch.object(audio, "which", path_with("pacat")):
            self.assertEqual(audio.build_play_cmd(cfg, 22050),
                             ["tee", "sink0-22050.raw"])

    def test_an_override_is_never_a_shell_string(self) -> None:
        for key, build in (("audio.capture_cmd",
                            lambda cfg: audio.build_capture_cmd(cfg)),
                           ("audio.play_cmd",
                            lambda cfg: audio.build_play_cmd(cfg, 16000))):
            with self.subTest(key=key):
                leaf = key.split(".")[1]
                with self.assertRaises(audio.AudioError) as caught:
                    build({"audio": {leaf: "cat fixture.raw | tee out"}})
                self.assertIn("never through a shell", str(caught.exception))

    def test_an_override_naming_no_program_is_refused(self) -> None:
        for bad in ([], [""]):
            with self.subTest(override=bad):
                with self.assertRaises(audio.AudioError) as caught:
                    audio.build_capture_cmd({"audio": {"capture_cmd": bad}})
                self.assertIn("names no program", str(caught.exception))


class Validation(unittest.TestCase):
    """Config values become argv, so they are checked rather than passed on."""

    def test_a_device_token_is_held_to_an_alphabet(self) -> None:
        for bad in ("mic; rm -rf /", "a b", "x" * 129, "$(id)", "../x"):
            with self.subTest(device=bad):
                with self.assertRaises(audio.AudioError) as caught:
                    audio.build_capture_cmd({"audio": {"device_in": bad}})
                self.assertIn("device name", str(caught.exception))

    def test_an_empty_device_reads_as_the_default(self) -> None:
        with mock.patch.object(audio, "which", path_with("parec")):
            for value in ("", None, "  default  "):
                with self.subTest(device=value):
                    cfg = {"audio": {"device_in": value}}
                    self.assertNotIn("-d", audio.build_capture_cmd(cfg))

    def test_an_unusable_rate_is_refused(self) -> None:
        for bad in ("fast", 0, -16000, None, [16000]):
            with self.subTest(rate=bad):
                with self.assertRaises(audio.AudioError) as caught:
                    audio.build_capture_cmd({"audio": {"rate": bad}})
                self.assertIn("audio.rate", str(caught.exception))

    def test_a_float_rate_that_is_whole_is_accepted(self) -> None:
        with mock.patch.object(audio, "which", path_with("parec")):
            cmd = audio.build_capture_cmd({"audio": {"rate": 16000.0}})
        self.assertIn("--rate=16000", cmd)


class Construction(unittest.TestCase):
    """Constructing a capture or a player touches no audio device."""

    def test_nothing_is_launched_before_start_or_play(self) -> None:
        cfg = {"audio": {"capture_cmd": ["cat"], "play_cmd": ["cat"]}}
        with no_launch():
            capture = audio.MicCapture(cfg)
            player = audio.Player(cfg)
            # An empty clip is silence, correctly rendered by no sink at all.
            player.play(b"", 22050)
            player.play(b"\x01", 22050)   # half a sample: nothing to play
            self.assertEqual(capture.frame_bytes, 640)
            self.assertFalse(player.playing)
            self.assertTrue(player.wait(0))

    def test_frame_geometry_follows_the_config(self) -> None:
        capture = audio.MicCapture({"audio": {"rate": 8000, "frame_ms": 30}})
        self.assertEqual(capture.rate, 8000)
        self.assertEqual(capture.frame_bytes, 8000 * 30 // 1000 * 2)

    def test_the_frozen_defaults_are_16_kHz_and_20_ms(self) -> None:
        capture = audio.MicCapture()
        self.assertEqual(capture.rate, audio.DEFAULT_RATE)
        self.assertEqual(capture.frame_bytes, 640)
        self.assertIsNone(capture.error)
        self.assertEqual(capture.overruns, 0)

    def test_a_frame_shorter_than_one_sample_is_refused(self) -> None:
        with self.assertRaises(audio.AudioError) as caught:
            audio.MicCapture({"audio": {"rate": 100, "frame_ms": 1}})
        self.assertIn("audio.frame_ms", str(caught.exception))

    def test_an_unusable_rate_is_refused_at_construction(self) -> None:
        with self.assertRaises(audio.AudioError):
            audio.MicCapture({"audio": {"rate": 0}})


class CaptureLifecycle(unittest.TestCase):
    """The click-to-talk guarantee: no audio exists before start()."""

    def test_read_before_start_is_refused(self) -> None:
        capture = audio.MicCapture({"audio": {"capture_cmd": ["cat"]}})
        with self.assertRaises(audio.AudioError) as caught:
            capture.read(timeout=0)
        message = str(caught.exception)
        self.assertIn("start()", message)
        self.assertIn("holds no audio", message)

    def test_stop_without_start_is_harmless(self) -> None:
        with no_launch():
            capture = audio.MicCapture({"audio": {"capture_cmd": ["cat"]}})
            capture.stop()
            capture.stop()

    def test_the_command_is_validated_before_any_launch(self) -> None:
        with no_launch():
            cfg = {"audio": {"capture_cmd": "cat fixture"}}
            capture = audio.MicCapture(cfg)
            with self.assertRaises(audio.AudioError) as caught:
                capture.start()
        self.assertIn("audio.capture_cmd", str(caught.exception))


class PlayerRefusals(unittest.TestCase):
    """Player rejects what it cannot play before it opens a sink."""

    def test_pcm_must_be_bytes(self) -> None:
        with no_launch():
            player = audio.Player({"audio": {"play_cmd": ["cat"]}})
            with self.assertRaises(audio.AudioError) as caught:
                player.play("hello", 22050)
        self.assertIn("s16le PCM bytes", str(caught.exception))

    def test_the_clip_rate_must_be_usable(self) -> None:
        with no_launch():
            player = audio.Player({"audio": {"play_cmd": ["cat"]}})
            with self.assertRaises(audio.AudioError):
                player.play(b"\x00\x00", 0)

    def test_play_after_close_is_refused(self) -> None:
        with no_launch():
            player = audio.Player({"audio": {"play_cmd": ["cat"]}})
            player.close()
            player.close()      # idempotent
            with self.assertRaises(audio.AudioError) as caught:
                player.play(b"\x00\x00", 22050)
        self.assertIn("Construct a new Player", str(caught.exception))

    def test_stop_on_an_idle_player_is_harmless(self) -> None:
        with no_launch():
            player = audio.Player({})
            player.stop()
            self.assertFalse(player.playing)
            self.assertIsNone(player.error)
            self.assertTrue(player.wait(0))


if __name__ == "__main__":
    unittest.main()
