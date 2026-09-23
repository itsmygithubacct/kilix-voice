"""Offline executable-boundary tests for Voice's explicit Qwen adapter."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from voicelib import models, protocol, tts
from voicelib.qwen_provider import QwenProviderTts, provider_binary


CLIENT = '''#!/usr/bin/env python3
import hashlib,json,os,struct,sys,time
args=sys.argv[1:]
if args[0]=='models':
    print(json.dumps({'models':[{'id':'qwen3-tts-0.6b-customvoice','installed':True,
      'capabilities':['named_voice'],'asset_authority':os.getenv('FAKE_AUTHORITY','kilix-content')}]}))
    raise SystemExit(0)
assert args[0]=='synthesize'
for required in ('--wav-stdout','--require-installed-asset','--model-id','--voice-id'):
    assert required in args
assert args[args.index('--model-id')+1]=='qwen3-tts-0.6b-customvoice'
assert args[args.index('--voice-id')+1]=='Vivian'
assert sys.stdin.buffer.read()==b'Hello'
if os.getenv('FAKE_SLEEP'):
    time.sleep(60)
failure=os.getenv('FAKE_REFUSAL')
if failure:
    print(f'KILIX_QWEN_TTS_REFUSAL [{failure}] speech request failed',file=sys.stderr)
    raise SystemExit(69)
pcm=b'\\x01\\x00'*2400
wav=struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',len(pcm)+36,b'WAVE',b'fmt ',
    16,1,1,24000,48000,2,16,b'data',len(pcm))+pcm
sys.stdout.buffer.write(wav)
print(json.dumps({'model_id':'qwen3-tts-0.6b-customvoice','seed':0,
    'audio':{'byte_length':len(wav),'sha256':
        '0'*64 if os.getenv('FAKE_BAD_DIGEST') else hashlib.sha256(wav).hexdigest()}}),file=sys.stderr)
'''


class QwenAdapterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="voice-qwen-client-")
        self.addCleanup(temp.cleanup)
        script = Path(temp.name) / "kilix-qwen-tts"
        script.write_text(CLIENT)
        script.chmod(0o700)
        binary = patch("voicelib.qwen_provider.provider_binary", return_value=str(script))
        binary.start()
        self.addCleanup(binary.stop)

    def test_explicit_selection_and_bound_audio(self):
        engine = tts.make_tts(model=models.QWEN_CUSTOMVOICE_MODEL)
        pcm, rate = engine.synth("Hello", budget=2)
        self.assertEqual((len(pcm), rate), (4800, 24000))
        self.assertEqual(engine.last_provenance.model, engine.model)
        self.assertEqual(engine.last_provenance.seed, 0)

    def test_unreceipted_runtime_is_refused(self):
        with patch.dict(os.environ, {"FAKE_AUTHORITY": "local-stage"}):
            with self.assertRaises(tts.TtsError) as caught:
                QwenProviderTts().check_available()
        self.assertIn("receipt-backed", str(caught.exception))

    def test_managed_client_is_discovered_without_daemon_restart(self):
        with patch.dict(os.environ, {"KILIX_QWEN_TTS": ""}), \
                patch("voicelib.qwen_provider.paths.data_dir", return_value="/data/voice"), \
                patch("voicelib.qwen_provider.util.which", side_effect=lambda path: path):
            self.assertEqual(provider_binary(),
                             "/data/voice/qwen-client/current/bin/kilix-qwen-tts")

    def test_unsupported_rate_and_voice(self):
        with self.assertRaises(tts.TtsUnsupported):
            QwenProviderTts(rate=170)
        with self.assertRaises(tts.TtsUnsupported):
            QwenProviderTts(voice="/tmp/voice")

    def test_busy_and_deadline_codes(self):
        for provider_code, expected in (("BUSY", protocol.ERR_BUSY),
                                        ("DEADLINE_EXCEEDED", protocol.ERR_DEADLINE)):
            with self.subTest(provider_code=provider_code):
                with patch.dict(os.environ, {"FAKE_REFUSAL": provider_code}):
                    with self.assertRaises(tts.TtsError) as caught:
                        QwenProviderTts().synth("Hello", budget=2)
                self.assertEqual(caught.exception.code, expected)

    def test_cancelled_before_synthesis_does_not_start_client(self):
        engine = QwenProviderTts()
        engine.cancel()
        self.assertEqual(engine.synth("Hello"), (b"", 24000))

    def test_cancel_kills_an_active_client_without_delivering_audio(self):
        engine = QwenProviderTts()
        result = []
        with patch.dict(os.environ, {"FAKE_SLEEP": "1"}):
            worker = threading.Thread(target=lambda: result.append(engine.synth("Hello")))
            worker.start()
            deadline = time.monotonic() + 2
            while engine._process is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(engine._process)
            engine.cancel()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [(b"", 24000)])

    def test_rejects_result_whose_audio_digest_does_not_match(self):
        with patch.dict(os.environ, {"FAKE_BAD_DIGEST": "1"}):
            with self.assertRaises(tts.TtsError) as caught:
                QwenProviderTts().synth("Hello")
        self.assertIn("invalid speech", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
