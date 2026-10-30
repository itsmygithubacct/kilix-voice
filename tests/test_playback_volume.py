"""Conversation gain reaches the sink and cannot affect another owner's turn."""
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest

from voicelib import audio, protocol
from voicelib.playback_volume import PlaybackVolume
from tests.test_owned_speech import Harness


class VolumeTests(unittest.TestCase):
    def test_full_range_without_announcement_cap(self):
        pcm = struct.pack('<5h', -32768, -10000, 0, 10000, 32767)
        gain = PlaybackVolume()
        self.assertIs(gain.apply(pcm), pcm)
        for level, expected in ((100, (-32768, -10000, 0, 10000, 32767)),
                                (50, (-16384, -5000, 0, 5000, 16383)),
                                (20, (-6553, -2000, 0, 2000, 6553)),
                                (0, (0, 0, 0, 0, 0))):
            gain.set(level)
            self.assertEqual(struct.unpack('<5h', gain.apply(pcm)), expected)

    def test_protocol_rejects_bad_gain_and_preserves_legacy_requests(self):
        for level in (-1, 101, True, 1.5, '20', None):
            with self.subTest(level=level):
                with self.assertRaises(ValueError):
                    PlaybackVolume(level)
                with self.assertRaises(protocol.ProtocolError):
                    protocol.validate_request(dict(op='owned-speak', owner='a'*32,
                        utterance='b'*32, text='hello', volume=level), '/tmp')
        request = protocol.validate_request(dict(op='speak', text='legacy'), '/tmp')
        self.assertNotIn('volume', request)

    def test_owner_turn_scope_and_idempotence(self):
        daemon = Harness()
        request = protocol.validate_request(dict(op='owned-speak', owner='a'*32,
            utterance='b'*32, text='hello', volume=100), '/tmp')
        first = daemon._op_owned(request)
        control = dict(op='owned-volume', owner='a'*32, turn=first['turn'], volume=0)
        for changes in ({'owner': 'c'*32}, {'turn': 'speak-'+'0'*16+'-999'}):
            reply = daemon._op_owned(protocol.validate_request(dict(control, **changes), '/tmp'))
            self.assertFalse(reply['applied'])
            self.assertEqual(daemon._speech.playback_volume.percent, 100)
        self.assertTrue(daemon._op_owned(protocol.validate_request(control, '/tmp'))['applied'])
        self.assertEqual(daemon._speech.playback_volume.percent, 0)
        self.assertEqual(daemon._op_owned(request)['turn'], first['turn'])
        self.assertEqual(daemon._op_owned(dict(request, volume=50))['code'], protocol.ERR_MALFORMED)
        daemon._cancel_speech()
        daemon._op_owned(dict(request, utterance='d'*32, volume=75))
        self.assertFalse(daemon._op_owned(protocol.validate_request(control, '/tmp'))['applied'])
        self.assertEqual(daemon._speech.playback_volume.percent, 75)

    def test_live_gain_changes_pcm_without_replaying_or_leaking_to_next_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sink = root/'sink.py'
            sink.write_text("import pathlib,sys,time\n"
                "root=pathlib.Path(sys.argv[1])\n"
                "first=sys.stdin.buffer.read(1024)\n"
                "(root/'started').write_bytes(first)\n"
                "limit=time.monotonic()+5\n"
                "while not (root/'release').exists() and time.monotonic()<limit: time.sleep(.005)\n"
                "(root/'pcm').write_bytes(first+sys.stdin.buffer.read())\n")
            player = audio.Player({'audio': {'play_cmd': [sys.executable, str(sink), str(root)]}})
            gain = PlaybackVolume(100)
            pcm = struct.pack('<h', 10000) * 100000
            try:
                player.play(pcm, 16000, volume=gain)
                deadline = time.monotonic()+3
                while not (root/'started').exists() and time.monotonic()<deadline:
                    time.sleep(.005)
                self.assertTrue((root/'started').exists())
                self.assertEqual((root/'started').read_bytes(), pcm[:1024])
                gain.set(0)
                (root/'release').touch()
                self.assertTrue(player.wait(5))
                self.assertIsNone(player.error)
                played = (root/'pcm').read_bytes()
                self.assertEqual(len(played), len(pcm))
                self.assertEqual(played[:1024], pcm[:1024])
                self.assertEqual(played[-32000:], bytes(32000))
                # The shared player's next unscoped clip retains full gain.
                player.play(pcm[:2048], 16000)
                self.assertTrue(player.wait(3))
                self.assertEqual((root/'pcm').read_bytes(), pcm[:2048])
            finally:
                player.close()


if __name__ == '__main__':
    unittest.main()
