"""System speech attenuates PCM independently of output backend/master gain."""
import contextlib
import io
import struct
import unittest
from unittest import mock

from voicelib import system_voice as V
from tests.test_tts_tool import load_tool


class VolumeTests(unittest.TestCase):
    def test_pcm_limit_and_quieter_choices(self):
        pcm = struct.pack('<5h', -32768, -10000, 0, 10000, 32767)
        for volume, expected in ((20, (-6553, -2000, 0, 2000, 6553)),
                                 (100, (-6553, -2000, 0, 2000, 6553)),
                                 (5, (-1638, -500, 0, 500, 1638)),
                                 (0, (0, 0, 0, 0, 0))):
            self.assertEqual(struct.unpack('<5h', V.quiet_pcm(pcm, volume)), expected)
        self.assertEqual(struct.unpack('<5h', pcm), (-32768, -10000, 0, 10000, 32767))

    def test_muted_clip_never_opens_output(self):
        player = mock.Mock()
        V.play_clip(player, (b'\x10\x27', 22050), 0)
        player.play.assert_not_called()
        player.wait.assert_not_called()

    def test_cli_routes_volume_and_rejects_out_of_range_or_unrelated(self):
        tool = load_tool()
        with mock.patch.object(V, 'run', return_value=0) as run:
            self.assertEqual(tool.main(['--system-voice', '--system-volume=5']), 0)
        run.assert_called_once_with('hello', volume=5)
        for argv in (['--system-volume=5'], ['--system-voice', '--system-volume=21'],
                     ['--system-voice', '--system-volume=-1'], ['--speak=x', '--system-volume=5']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                tool.main(argv)
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
