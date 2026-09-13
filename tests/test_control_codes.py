"""Every refusal on the control path carries the code for its cause.

R6 finding 3: well-formed requests refused by a handler or by _answer all
reached the caller as ERR_INTERNAL, "the daemon has a bug", and ERR_BUSY and
ERR_NOT_FOUND were used nowhere. These tests drive the real _dispatch and
_answer and assert only the code on the reply -- the prose is for a human and
may change; the code is the contract.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import socket
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_codes", importlib.machinery.SourceFileLoader(
        "kilix_voiced_codes", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import models, protocol, tts as tts_lib  # noqa: E402
from voicelib.arbiter import Arbiter  # noqa: E402


class _ControlFixture(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = os.path.realpath(self.tmp.name)

    def daemon(self):
        d = object.__new__(voiced.Daemon)
        d._session_dir = self.session
        d._lock = threading.RLock()
        d._cfg = {"stt": {"engine": "vosk"}}
        d._refresh_config = lambda: None
        d._touch = lambda: None
        d._warn = d._debug = lambda *a, **k: None
        d._next_turn_id = lambda kind: f"{kind}-9"
        d._speech = None
        d._player = None
        d._speech_error = ""
        d._arbiter = mock.Mock(listening=False, speaking=False)
        d._clear_dictation = lambda turn: None
        d._speech_chunks = lambda text: ["one"]
        return d

    def dispatch(self, d, message):
        return voiced.Daemon._dispatch(d, protocol.encode(message))

    def in_session(self, name):
        return os.path.join(self.session, name)


class HandlerRefusalCodesTestCase(_ControlFixture):

    def test_a_second_dictate_is_busy(self) -> None:
        d = self.daemon()
        d._arbiter = mock.Mock(listening=True)
        reply = self.dispatch(d, {"op": "dictate",
                                  "sock": self.in_session("dictate-1.sock")})
        self.assertEqual(reply["code"], protocol.ERR_BUSY, reply)

    def test_dictation_switched_off_is_unavailable(self) -> None:
        d = self.daemon()
        d._cfg = {"stt": {"engine": "off"}}
        reply = self.dispatch(d, {"op": "dictate",
                                  "sock": self.in_session("dictate-1.sock")})
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)

    def test_speak_while_dictation_holds_the_microphone_is_busy(self) -> None:
        d = self.daemon()
        d._arbiter = Arbiter(self.session)
        d._arbiter.begin_listen("listen-1")
        with mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: mock.Mock(model="m", voice="v",
                                                         rate=170)):
            reply = self.dispatch(d, {"op": "speak", "text": "hello"})
        self.assertEqual(reply["code"], protocol.ERR_BUSY, reply)

    def test_a_successor_speak_in_the_hand_over_window_is_busy(self) -> None:
        # R6 property 3: the old worker has already cleared _speech but has
        # not yet called arbiter.end_speech, so the speakers are still claimed.
        d = self.daemon()
        d._arbiter = Arbiter(self.session)
        d._arbiter.begin_speech("speak-1")
        d._speech = None
        with mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: mock.Mock(model="m", voice="v",
                                                         rate=170)):
            reply = self.dispatch(d, {"op": "speak", "text": "hello"})
        self.assertEqual(reply["code"], protocol.ERR_BUSY, reply)

    def test_a_dictation_socket_that_does_not_exist_is_not_found(self) -> None:
        d = self.daemon()
        reply = self.dispatch(d, {"op": "dictate",
                                  "sock": self.in_session("dictate-7.sock")})
        self.assertEqual(reply["code"], protocol.ERR_NOT_FOUND, reply)

    def test_a_regular_file_where_the_socket_should_be_is_not_found(self) -> None:
        path = self.in_session("dictate-7.sock")
        with open(path, "w") as handle:
            handle.write("not a socket")
        d = self.daemon()
        reply = self.dispatch(d, {"op": "dictate", "sock": path})
        self.assertEqual(reply["code"], protocol.ERR_NOT_FOUND, reply)

    def test_a_chunk_socket_nobody_listens_on_is_not_found(self) -> None:
        path = self.in_session("chunks-1.sock")
        dead = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        dead.bind(path)
        dead.close()                      # the file stays; nothing accepts
        d = self.daemon()
        with mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: mock.Mock(model="m", voice="v",
                                                         rate=170)):
            reply = self.dispatch(d, {"op": "speak", "text": "hello",
                                      "chunk_sock": path})
        self.assertEqual(reply["code"], protocol.ERR_NOT_FOUND, reply)

    def test_a_worker_thread_the_os_will_not_start_is_unavailable(self) -> None:
        class _Unstartable:
            def __init__(self, *a, **k):
                pass

            def start(self):
                raise RuntimeError("can't start new thread")

        d = self.daemon()
        with mock.patch.object(voiced, "_connect_dictation",
                               lambda path: mock.Mock()), \
             mock.patch.object(voiced.threading, "Thread", _Unstartable):
            reply = self.dispatch(d, {"op": "dictate",
                                      "sock": self.in_session("dictate-1.sock")})
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)

    def test_a_barge_in_hook_that_fails_is_unavailable(self) -> None:
        def cannot_silence():
            raise OSError("the sink will not die")

        d = self.daemon()
        d._arbiter = Arbiter(self.session, cancel_speech=cannot_silence)
        d._arbiter.begin_speech("speak-1")
        with mock.patch.object(voiced, "_connect_dictation",
                               lambda path: mock.Mock()):
            reply = self.dispatch(d, {"op": "dictate",
                                      "sock": self.in_session("dictate-1.sock")})
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)

    def test_a_voice_the_piper_model_cannot_use_is_unsupported(self) -> None:
        d = self.daemon()
        d._cfg = {}
        reply = self.dispatch(d, {"op": "speak", "text": "hello",
                                  "model": models.PIPER_KRISTIN_MODEL,
                                  "voice": "en-us"})
        self.assertEqual(reply["code"], protocol.ERR_UNSUPPORTED, reply)

    def test_piper_not_installed_with_no_deadline_is_unavailable(self) -> None:
        d = self.daemon()
        d._cfg = {}
        missing = os.path.join(self.session, "no-such-kilix-piper-tts")
        with mock.patch.dict(os.environ, {tts_lib.PIPER_ENV_COMMAND: missing}):
            reply = self.dispatch(d, {"op": "speak", "text": "hello",
                                      "model": models.PIPER_KRISTIN_MODEL,
                                      "rate": 170})
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE, reply)
        self.assertIn("not installed", reply["error"])

    def test_a_handler_bug_is_still_internal(self) -> None:      # control
        d = self.daemon()
        d._op_status = mock.Mock(side_effect=KeyError("tts"))
        reply = self.dispatch(d, {"op": "status"})
        self.assertEqual(reply["code"], protocol.ERR_INTERNAL, reply)


class AnswerRefusalCodesTestCase(_ControlFixture):

    def test_a_connection_from_another_uid_is_denied(self) -> None:
        d = self.daemon()
        d._peer_uid = lambda conn: os.geteuid() + 1
        reply = voiced.Daemon._answer(d, mock.Mock())
        self.assertEqual(reply["code"], protocol.ERR_DENIED, reply)

    def test_an_oversized_request_is_too_large(self) -> None:
        d = self.daemon()
        d._peer_uid = lambda conn: os.geteuid()
        conn = mock.Mock()
        conn.recv.return_value = b"x" * (voiced.MAX_REQUEST_BYTES + 1)
        reply = voiced.Daemon._answer(d, conn)
        self.assertEqual(reply["code"], protocol.ERR_TOO_LARGE, reply)


class CarriedCodeTestCase(unittest.TestCase):
    """The one adapter both paths share."""

    def test_a_valid_carried_code_is_kept(self) -> None:
        self.assertEqual(voiced._carried_code(voiced.DaemonBusy("x"),
                                              protocol.ERR_INTERNAL),
                         protocol.ERR_BUSY)

    def test_a_code_outside_the_vocabulary_falls_back_to_the_default(self) -> None:
        error = RuntimeError("x")
        error.code = "not-a-code"
        self.assertEqual(voiced._carried_code(error, protocol.ERR_UNAVAILABLE),
                         protocol.ERR_UNAVAILABLE)

    def test_the_worker_and_the_control_path_agree_on_every_family(self) -> None:
        # The R5 finding A shape: one condition, two codes on two paths. These
        # are the refusal families a dictation turn can raise; ArbiterError and
        # TtsError cannot arise inside a dictation worker, so they have no
        # second path to disagree with. SettingsError was the one that did:
        # _record reads settings every turn and the worker coded it internal.
        families = (voiced.DaemonError("x"), voiced.DaemonBusy("x"),
                    voiced.ConsentDenied("x"), voiced.stt_lib.SttError("x"),
                    voiced.audio.AudioError("x"),
                    voiced.settings.SettingsError("x"),
                    voiced.paths.PathError("x"))
        for error in families:
            with self.subTest(family=type(error).__name__):
                sent = []
                d = object.__new__(voiced.Daemon)
                d._warn = lambda *a, **k: None
                d._send = lambda receiver, msg: sent.append(msg) or True
                d._clear_dictation = lambda turn: None
                d._touch = lambda: None
                d._dictate = mock.Mock(side_effect=error)
                voiced.Daemon._run_dictation(
                    d, voiced._DictationTurn("listen-1", mock.Mock()))
                self.assertEqual(len(sent), 1, sent)
                control = mock.Mock(side_effect=error)
                c = object.__new__(voiced.Daemon)
                c._session_dir = "/tmp"
                c._touch = lambda: None
                c._op_stop_speech = control
                reply = voiced.Daemon._dispatch(
                    c, protocol.encode({"op": "stop-speech"}))
                self.assertEqual(reply["code"], sent[0]["code"])


class ValidationRefusalCodesTestCase(_ControlFixture):
    """MALFORMED-GAP and V08: a caller's request shape is never `internal`."""

    def reply(self, raw):
        d = object.__new__(voiced.Daemon)
        d._session_dir = self.session
        d._touch = lambda: None
        return voiced.Daemon._dispatch(d, raw)

    def test_each_refusal_carries_the_code_for_its_cause(self) -> None:
        outside = os.path.join(os.path.dirname(self.session), "elsewhere.sock")
        table = (
            ({"op": "speak", "text": "hi", "deadline_ms": 0}, protocol.ERR_MALFORMED),
            ({"op": "speak", "text": "hi", "deadline_ms": "x"}, protocol.ERR_MALFORMED),
            ({}, protocol.ERR_MALFORMED),
            ({"op": 5}, protocol.ERR_MALFORMED),
            ({"op": "nonsense"}, protocol.ERR_UNSUPPORTED),
            ({"op": "speak"}, protocol.ERR_MALFORMED),
            ({"op": "status", "v": "2"}, protocol.ERR_UNSUPPORTED),
            ({"op": "status", "v": "0.9"}, protocol.ERR_UNSUPPORTED),
            ({"op": "status", "v": "11.1"}, protocol.ERR_UNSUPPORTED),
            ({"op": "status", "v": "abc"}, protocol.ERR_MALFORMED),
            ({"op": "dictate", "sock": outside}, protocol.ERR_MALFORMED),
        )
        for message, code in table:
            with self.subTest(message=message):
                reply = self.reply(protocol.encode(message))
                self.assertIs(reply["ok"], False, reply)
                self.assertEqual(reply["code"], code, reply)

    def test_frames_that_are_not_requests_are_malformed(self) -> None:
        for raw in (b"\xff", b"not json", b"[1]"):
            with self.subTest(raw=raw):
                self.assertEqual(self.reply(raw)["code"], protocol.ERR_MALFORMED)

    def test_an_oversized_frame_given_to_dispatch_is_too_large(self) -> None:
        reply = self.reply(b"x" * (protocol.MAX_REQUEST_BYTES + 1))
        self.assertEqual(reply["code"], protocol.ERR_TOO_LARGE, reply)

    def test_an_incompatible_major_is_exactly_a_protocol_error(self) -> None:
        # The frozen V08 vector compares the refusal's type exactly, so the
        # code must ride on the instance and the type must not change.
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.validate_request({"op": "status", "v": "2"}, self.session)
        self.assertIs(type(caught.exception), protocol.ProtocolError)
        self.assertEqual(caught.exception.code, protocol.ERR_UNSUPPORTED)

    def test_an_uncoded_protocol_error_in_the_worker_stays_unavailable(self) -> None:
        # Why the code is not on the ProtocolError base: a descriptor that
        # fails to build inside a dictation turn is not the caller's malformed
        # request, and R4 confirmed `unavailable` for that arm.
        sent = []
        d = object.__new__(voiced.Daemon)
        d._warn = lambda *a, **k: None
        d._send = lambda receiver, msg: sent.append(msg) or True
        d._clear_dictation = lambda turn: None
        d._touch = lambda: None
        d._dictate = mock.Mock(side_effect=protocol.ProtocolError("bad segment"))
        voiced.Daemon._run_dictation(d, voiced._DictationTurn("listen-1", mock.Mock()))
        self.assertEqual([m.get("code") for m in sent], [protocol.ERR_UNAVAILABLE])

    def test_a_code_outside_the_vocabulary_cannot_be_attached(self) -> None:
        with self.assertRaises(ValueError):
            protocol.ProtocolError("x", code="oops")


def _load_tool(name, filename):
    spec = importlib.util.spec_from_loader(
        name, importlib.machinery.SourceFileLoader(name, os.path.join(ROOT, filename)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FirstPartyClientsDeclareTheirVersionTestCase(unittest.TestCase):
    """The daemon's own tools say which contract they were written against."""

    def test_kilix_tts_declares_v_on_status_and_stop(self) -> None:
        tool = _load_tool("kilix_tts_codes", "kilix-tts")
        sent = []

        def control(request, *args, **kwargs):
            sent.append(request)
            if request["op"] == protocol.OP_STATUS:
                raise RuntimeError("captured")
            return {"ok": True, "stopped": True}

        with mock.patch.object(tool, "control", control):
            with self.assertRaises(RuntimeError):
                tool.probe()
            tool.stop_speech()
            tool._compensate_speech()
        self.assertEqual([r["op"] for r in sent],
                         [protocol.OP_STATUS, protocol.OP_STOP_SPEECH,
                          protocol.OP_STOP_SPEECH])
        for request in sent:
            self.assertEqual(request["v"], protocol.PROTOCOL_VERSION)

    def test_kilix_stt_declares_v_on_status(self) -> None:
        tool = _load_tool("kilix_stt_codes", "kilix-stt")
        sent = []

        class Endpoint:
            def __init__(self, *a, **k): pass
            def settimeout(self, timeout): pass
            def connect(self, path): pass
            def send(self, data): sent.append(protocol.decode(data)); return len(data)
            def recv(self, size): return protocol.encode(protocol.reply_ok("", status={}))
            def close(self): pass

        with mock.patch.object(tool.socket, "socket", Endpoint):
            self.assertEqual(tool.daemon_status(), {})
        self.assertEqual([r["v"] for r in sent], [protocol.PROTOCOL_VERSION])


if __name__ == "__main__":
    unittest.main()
