"""Owned speech: owner isolation, idempotent utterances and lease expiry.

`OwnedSpeechTests` pins the bookkeeping with a stub daemon. `OwnedDaemonTests`
runs the real kilix-voiced with a synthetic synthesiser and sink, so the
lifecycle it asserts is the one the worker threads actually settle.
"""
import pathlib
import socket
import sys
import threading
import time
import json
import unittest
from types import SimpleNamespace

from tests import test_daemon as daemon_tests
from voicelib.owned import OwnedTurns
from voicelib import jobs, protocol


class Harness(OwnedTurns):
    def __init__(self):
        self._lock=threading.RLock()
        self._speech=self._dictation=None
        self._arbiter=SimpleNamespace(listening=False,speaking=False)
        self._player=SimpleNamespace(playing=False)
        self.starts=0
        self._turns=0
    def _next_turn_id(self, kind):
        self._turns+=1
        return f'{kind}-{self._turns}'
    def _op_speak(self, request, claim=None):
        self.starts+=1
        turn=SimpleNamespace(id=self._next_turn_id('speak'),outcome=None)
        if claim is not None: claim(turn)
        self._speech=turn
        self._arbiter.speaking=True
        return {'ok':True,'turn':turn.id}
    def _cancel_speech(self):
        self._speech.outcome=SimpleNamespace(outcome=jobs.OUTCOME_CANCELLED,message='')
        self._speech=None
        self._arbiter.speaking=False

class OwnedSpeechTests(unittest.TestCase):
    def request(self, **changes):
        request=dict(op='owned-speak',owner='a'*32,utterance='b'*32,text='Hello',id=1)
        request.update(changes)
        return protocol.validate_request(request,"/tmp")
    def test_retry_is_idempotent_and_different_text_refused(self):
        d=Harness(); first=d._op_owned(self.request())
        self.assertTrue(first['ok'])
        from tests.test_control_codes import voiced
        self.assertTrue(voiced._guard_reply(first)['ok'])
        self.assertIn('playing',first)
        self.assertEqual(d._op_owned(self.request())['turn'],first['turn'])
        self.assertEqual(d.starts,1)
        conflict=d._op_owned(self.request(text='Different'))
        self.assertFalse(conflict['ok'])
        self.assertEqual(conflict['code'],protocol.ERR_MALFORMED)
    def test_other_owner_cannot_inspect_or_stop(self):
        d=Harness(); turn=d._op_owned(self.request())['turn']
        self.assertEqual(d._op_owned(self.request(op='owned-status',owner='c'*32))['state'],'unknown')
        self.assertFalse(d._op_owned(self.request(op='owned-stop',owner='c'*32,turn=turn))['stopped'])
        self.assertFalse(d._op_owned(self.request(owner='c'*32))['ok'])
        self.assertIsNotNone(d._speech)
    def test_lease_expiry_is_reported_as_expired(self):
        d=Harness();d._op_owned(self.request())
        d._speech.lease_until=time.monotonic()-1
        d._owned_expire()
        self.assertIsNone(d._speech)
        self.assertEqual(d._op_owned(self.request(op='owned-status'))['state'],'expired')
    def test_heartbeat_renews_and_wrong_turn_does_not_cancel(self):
        d=Harness();turn=d._op_owned(self.request())['turn']
        d._speech.lease_until=0
        d._op_owned(self.request(op='owned-status'))
        d._owned_expire()
        self.assertIsNotNone(d._speech)
        self.assertFalse(d._op_owned(self.request(op='owned-stop',turn='speak-'+'0'*16+'-999'))['stopped'])
        self.assertTrue(d._op_owned(self.request(op='owned-stop',turn=turn))['stopped'])
        self.assertEqual(d._op_owned(self.request(op='owned-status'))['state'],'cancelled')
    def test_turn_names_the_daemon_instance(self):
        d=Harness();turn=d._op_owned(self.request())['turn']
        self.assertRegex(turn,r'^speak-[0-9a-f]{16}-1$')
        restarted=Harness();restarted._op_owned(self.request())
        # The same owner's stop from before a restart cannot hit the new turn.
        self.assertFalse(restarted._op_owned(self.request(op='owned-stop',turn=turn))['stopped'])
        self.assertIsNotNone(restarted._speech)
    def test_tokens_and_limits(self):
        for changes in ({'owner':'bad'},{'utterance':'bad'},{'text':'x'*4097}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(protocol.ProtocolError): self.request(**changes)
        with self.assertRaises(protocol.ProtocolError):
            self.request(op='owned-stop',turn='speak-1')


class OwnedDaemonTests(unittest.TestCase):
    """The real daemon; synthetic PCM and a controllable fake sink."""

    setUp = daemon_tests.DaemonTestCase.setUp
    tearDown = daemon_tests.DaemonTestCase.tearDown
    _stop_daemon = daemon_tests.DaemonTestCase._stop_daemon
    _environment = daemon_tests.DaemonTestCase._environment
    _log_tail = daemon_tests.DaemonTestCase._log_tail
    _wait_until_serving = daemon_tests.DaemonTestCase._wait_until_serving
    _send = daemon_tests.DaemonTestCase._send
    request = daemon_tests.DaemonTestCase.request

    def _fixture(self, *, fail=False, delay=0.8):
        synthesiser = pathlib.Path(self.nowhere) / "espeak-ng"
        synthesiser.write_text(
            f"#!{sys.executable}\nimport io, sys, wave\n"
            "sys.stdin.buffer.read()\noutput = io.BytesIO()\n"
            "with wave.open(output, 'wb') as wav:\n"
            "    wav.setnchannels(1)\n    wav.setsampwidth(2)\n"
            "    wav.setframerate(22050)\n    wav.writeframes(b'\\x01\\x00' * 1000)\n"
            "sys.stdout.buffer.write(output.getvalue())\n", encoding="utf-8")
        synthesiser.chmod(0o755)
        self.play_log = pathlib.Path(self.root) / "play-started"
        sink = pathlib.Path(self.nowhere) / "pacat"
        sink.write_text(
            f"#!{sys.executable}\nimport sys, time\nsys.stdin.buffer.read()\n"
            f"with open({str(self.play_log)!r}, 'a') as log: log.write('played\\n')\n"
            + ("raise SystemExit(23)\n" if fail else f"time.sleep({delay!r})\n"),
            encoding="utf-8")
        sink.chmod(0o755)

    def _speak(self, owner="a" * 32, utterance="b" * 32, text="owned phrase"):
        return self.request({"op": "owned-speak", "owner": owner,
                             "utterance": utterance, "text": text, "model": "espeak"})

    def _status(self, owner="a" * 32, utterance="b" * 32):
        return self.request({"op": "owned-status", "owner": owner, "utterance": utterance})

    def _wait(self, states, utterance="b" * 32):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            reply = self._status(utterance=utterance)
            if reply.get("state") in states:
                return reply
            time.sleep(0.01)
        self.fail(f"Owned turn did not reach {states}: {reply}")

    def test_lifecycle_and_capability(self):
        self._fixture()
        status = self.request({"op": "status"})["status"]
        self.assertIn("owned-speech/v1", status["capabilities"])
        accepted = self._speak()
        self.assertTrue(accepted["ok"], accepted)
        self.assertRegex(accepted["turn"], r"^speak-[0-9a-f]{16}-[0-9]+$")
        active = self._wait(("speaking",))
        self.assertEqual(active["turn"], accepted["turn"])
        self.assertTrue(active["playing"])
        done = self._wait(("completed",))
        self.assertFalse(done["playing"])
        self.assertEqual(done["detail"], "")

    def test_admission_does_not_replace_any_current_turn(self):
        self._fixture()
        first = self._speak()
        busy = self._speak(owner="c" * 32)
        self.assertFalse(busy["ok"])
        self.assertEqual(busy["code"], protocol.ERR_BUSY)
        self.assertEqual(self._status()["turn"], first["turn"])
        self.assertEqual(self._status(owner="c" * 32)["state"], "unknown")
        # Nor does it take over a legacy read.
        self._wait(("completed",))
        self.assertTrue(self.request({"op": "speak", "text": "legacy speech"})["ok"])
        self.assertEqual(self._speak(utterance="d" * 32)["code"], protocol.ERR_BUSY)

    def test_stop_requires_owner_and_current_turn(self):
        self._fixture()
        first = self._speak()
        wrong = self.request({"op": "owned-stop", "owner": "c" * 32, "turn": first["turn"]})
        self.assertFalse(wrong["stopped"])
        stopped = self.request({"op": "owned-stop", "owner": "a" * 32, "turn": first["turn"]})
        self.assertTrue(stopped["stopped"])
        self._wait(("cancelled",))
        second = self._speak(utterance="d" * 32)
        self.assertTrue(second["ok"], second)
        stale = self.request({"op": "owned-stop", "owner": "a" * 32, "turn": first["turn"]})
        self.assertFalse(stale["stopped"])
        self.assertEqual(self._status(utterance="d" * 32)["turn"], second["turn"])
        self.assertIn(self._status(utterance="d" * 32)["state"], ("preparing", "speaking"))

    def test_retry_is_not_replayed_after_completion(self):
        self._fixture()
        first = self._speak()
        self.assertEqual(self._speak()["turn"], first["turn"])
        self._wait(("completed",))
        retry = self._speak()
        self.assertEqual(retry["turn"], first["turn"])
        self.assertEqual(retry["state"], "completed")
        self.assertEqual(self.play_log.read_text().splitlines(), ["played"])
        conflict = self._speak(text="different phrase")
        self.assertFalse(conflict["ok"])
        self.assertEqual(conflict["code"], protocol.ERR_MALFORMED)

    def test_lost_ack_can_be_reconciled(self):
        self._fixture()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.connect(self.control)
        client.send(json.dumps({"op": "owned-speak", "owner": "a" * 32,
                                "utterance": "b" * 32, "text": "owned phrase",
                                "model": "espeak"}).encode())
        client.close()  # Deliberately lose the acceptance reply.
        active = self._wait(("preparing", "speaking", "completed"))
        self.assertEqual(self._speak()["turn"], active["turn"])
        self._wait(("completed",))
        self.assertEqual(self.play_log.read_text().splitlines(), ["played"])

    def test_legacy_replacement_is_reported_as_superseded(self):
        self._fixture()
        first = self._speak()
        self.assertTrue(self.request({"op": "speak", "text": "legacy speech"})["ok"])
        self.assertEqual(self._wait(("superseded", "cancelled"))["state"], "superseded")
        stale = self.request({"op": "owned-stop", "owner": "a" * 32, "turn": first["turn"]})
        self.assertFalse(stale["stopped"])
        self.assertTrue(self.request({"op": "status"})["status"]["speaking"])

    def test_async_failure_carries_its_detail(self):
        self._fixture(fail=True)
        self.assertTrue(self._speak()["ok"])
        failed = self._wait(("failed",))
        self.assertIn("status 23", failed["detail"])
        self.assertFalse(failed["playing"])

    def test_tokens_are_strict(self):
        for owner in ("", "x" * 32, "A" * 32, [], 0):
            reply = self.request({"op": "owned-speak", "owner": owner,
                                  "utterance": "b" * 32, "text": "hello"})
            self.assertFalse(reply["ok"], reply)
        for turn in ("speak-1", "speak-" + "0" * 16):
            reply = self.request({"op": "owned-stop", "owner": "a" * 32, "turn": turn})
            self.assertFalse(reply["ok"], reply)
        self.assertEqual(self._status()["state"], "unknown")

    def test_client_loss_expires_the_speech_lease(self):
        self._fixture(delay=10)
        self.assertTrue(self._speak()["ok"])
        started = time.monotonic()
        while time.monotonic() - started < 7:
            # Global status does not renew an owner's lease.
            if not self.request({"op": "status"})["status"]["speaking"]:
                break
            time.sleep(0.03)
        self.assertEqual(self._wait(("expired", "cancelled"))["state"], "expired")
        self.assertLess(time.monotonic() - started, 6.5)


class RecommendTests(unittest.TestCase):
    def test_recommend_is_read_only_and_preserves_tier_json(self):
        import contextlib
        import io
        from unittest.mock import patch
        from tests.test_tts_tool import load_tool
        tool=load_tool()
        with patch.object(tool.sizing,'recommend',return_value={'fixture':True}) as recommend, \
             patch.object(tool.sizing,'print_report') as output, \
             patch.object(tool.tts_lib,'piper_status',return_value=(True,'')), \
             patch.object(tool.tts_lib,'espeak_binary',return_value='/fixture/espeak'):
            self.assertEqual(tool.main(['--recommend','--json']),0)
            self.assertEqual(recommend.call_args.args[0],'tts')
            self.assertTrue(recommend.call_args.args[1]['piper-en-us-kristin-medium'])
            output.assert_called_once_with({'fixture':True},as_json=True)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.main(['--recommend','--speak','hello'])


from tests.test_terminal_outcomes import FOUR, BIG, _LiveTurnsFixture, wait_until


class OwnedAdmissionRace(_LiveTurnsFixture):
    """Engine preparation runs outside the lock; admission must re-check."""

    def test_audio_started_during_preparation_refuses_the_owned_turn(self):
        daemon = self.daemon
        self.tts_sizes = [BIG]
        self.flag("hold", True)
        self.addCleanup(self.flag, "hold", False)
        prepare = daemon._refresh_config
        legacy = {}

        def racing_prepare():
            prepare()
            daemon._refresh_config = prepare
            legacy.update(self.call(op="speak", text=FOUR))

        daemon._refresh_config = racing_prepare
        owned = self.call(op="owned-speak", owner="a" * 32, utterance="b" * 32, text=FOUR)
        self.assertTrue(legacy.get("ok"), legacy)
        self.assertEqual(owned.get("code"), protocol.ERR_BUSY, owned)
        self.assertEqual(daemon._speech.id, legacy["turn"])
        self.assertEqual(self.call(op="owned-status", owner="a" * 32,
                                   utterance="b" * 32)["state"], "unknown")
