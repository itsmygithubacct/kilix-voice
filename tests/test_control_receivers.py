"""A receiver socket is never the daemon's own control socket.

Wave-1 observation: {op: dictate, sock: <the daemon's control.sock>} was
accepted. The path is inside the session directory, so containment let it
through, and the worker then connected to the daemon's own listener and sent
its datagrams there. The same held for speak's chunk_sock. The refusal compares
file identity, (st_dev, st_ino), so a hard link or a symlink naming the same
listener under another name is refused too.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import shutil
import socket
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_loader(
    "kilix_voiced_receivers", importlib.machinery.SourceFileLoader(
        "kilix_voiced_receivers", os.path.join(ROOT, "kilix-voiced")))
voiced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voiced)

from voicelib import protocol  # noqa: E402

from tests.livedaemon import LiveDaemonTestCase  # noqa: E402


def _listener(path: str) -> socket.socket:
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    endpoint.bind(path)
    endpoint.listen(4)
    endpoint.settimeout(0.2)
    return endpoint


class OwnControlSocketTestCase(unittest.TestCase):
    """The handlers, on a daemon whose control socket is a real bound listener."""

    def setUp(self) -> None:
        self.session = os.path.realpath(tempfile.mkdtemp(prefix="kv-rcv-", dir="/tmp"))
        self.addCleanup(shutil.rmtree, self.session, True)
        self.control = os.path.join(self.session, "control.sock")
        self.listener = _listener(self.control)
        self.addCleanup(self.listener.close)
        self.started = []

    def daemon(self):
        info = os.stat(self.control)
        d = object.__new__(voiced.Daemon)
        d._session_dir = self.session
        d._socket_path = self.control
        d._socket_dev, d._socket_ino = info.st_dev, info.st_ino
        d._lock = threading.RLock()
        d._cfg = {"stt": {"engine": "vosk"}}
        d._refresh_config = lambda: None
        d._touch = lambda: None
        d._warn = d._debug = lambda *a, **k: None
        d._next_turn_id = lambda kind: f"{kind}-9"
        d._speech = None
        d._dictation = None
        d._player = None
        d._speech_error = ""
        d._arbiter = mock.Mock(listening=False, speaking=False)
        d._clear_dictation = lambda turn: None
        d._speech_chunks = lambda text: ["one"]
        d._cancel_speech = lambda: None
        d._start = lambda thread, undo: self.started.append(thread.name)
        return d

    def dispatch(self, d, message):
        with mock.patch.object(voiced.tts_lib, "make_tts",
                               lambda *a, **k: mock.Mock(model="m", voice="v", rate=170)):
            return voiced.Daemon._dispatch(d, protocol.encode(message))

    def assert_refused_and_nothing_connected(self, reply) -> None:
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], protocol.ERR_MALFORMED, reply)
        self.assertIn("control socket", reply["error"])
        with self.assertRaises(socket.timeout):
            conn, _ = self.listener.accept()
            conn.close()
        self.assertEqual(self.started, [])

    def test_dictate_naming_the_control_socket_is_refused(self) -> None:
        reply = self.dispatch(self.daemon(), {"op": "dictate", "sock": self.control})
        self.assert_refused_and_nothing_connected(reply)

    def test_a_hard_link_to_the_control_socket_is_refused(self) -> None:
        link = os.path.join(self.session, "dictate-3.sock")
        os.link(self.control, link)
        reply = self.dispatch(self.daemon(), {"op": "dictate", "sock": link})
        self.assert_refused_and_nothing_connected(reply)

    def test_a_symlink_to_the_control_socket_is_refused(self) -> None:
        link = os.path.join(self.session, "dictate-4.sock")
        os.symlink(self.control, link)
        reply = self.dispatch(self.daemon(), {"op": "dictate", "sock": link})
        self.assert_refused_and_nothing_connected(reply)

    def test_chunk_sock_naming_the_control_socket_is_refused(self) -> None:
        reply = self.dispatch(self.daemon(), {"op": "speak", "text": "hello",
                                              "chunk_sock": self.control})
        self.assert_refused_and_nothing_connected(reply)

    def test_chunk_sock_through_a_hard_link_is_refused(self) -> None:
        link = os.path.join(self.session, "chunks-1.sock")
        os.link(self.control, link)
        reply = self.dispatch(self.daemon(), {"op": "speak", "text": "hello",
                                              "chunk_sock": link})
        self.assert_refused_and_nothing_connected(reply)

    def test_a_real_receiver_is_still_connected(self) -> None:          # control
        for op, field, name in (("dictate", "sock", "dictate-5.sock"),
                                ("speak", "chunk_sock", "chunks-5.sock")):
            with self.subTest(op=op):
                path = os.path.join(self.session, name)
                receiver = _listener(path)
                self.addCleanup(receiver.close)
                d = self.daemon()
                message = {"op": op, field: path}
                if op == "speak":
                    message["text"] = "hello"
                reply = self.dispatch(d, message)
                self.assertIs(reply["ok"], True, reply)
                conn, _ = receiver.accept()
                conn.close()
                turn = d._dictation if op == "dictate" else d._speech
                turn.receiver.close()
        self.assertEqual(len(self.started), 2)

    # W1-R3 residual and the dangling-symlink class: a name that is not UTF-8
    # decodes to lone surrogates, and a refusal quoting it could not be
    # encoded, so the caller was told `internal` instead of the refusal.

    def assert_sent_as(self, reply: dict, code: str) -> dict:
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], code, reply)
        # What _accept_one sends: the reply as built, never its internal stand-in.
        return protocol.decode(protocol.encode(reply, limit=protocol.MAX_REPLY_BYTES))

    def receiver_ops(self, prefix: str):
        for op, field in (("dictate", "sock"), ("speak", "chunk_sock")):
            link = os.path.join(self.session, f"{prefix}-{op}.sock")
            message = {"op": op, field: link}
            if op == "speak":
                message["text"] = "hello"
            yield op, link, message

    def test_a_symlink_to_a_non_utf8_link_of_the_control_socket_is_malformed(self) -> None:
        hidden = os.path.join(os.fsencode(self.session), b"\xff")
        os.link(os.fsencode(self.control), hidden)
        for op, link, message in self.receiver_ops("hidden"):
            with self.subTest(op=op):
                os.symlink(hidden, os.fsencode(link))
                sent = self.assert_sent_as(self.dispatch(self.daemon(), message),
                                           protocol.ERR_MALFORMED)
                self.assertIn("control socket", sent["error"])
                self.assertIn("\\udcff", sent["error"])       # the name, escaped
        with self.assertRaises(socket.timeout):
            conn, _ = self.listener.accept()
            conn.close()
        self.assertEqual(self.started, [])

    def test_a_non_utf8_name_that_is_missing_or_no_socket_is_not_found(self) -> None:
        plain = os.path.join(os.fsencode(self.session), b"file-\xfd")
        open(plain, "wb").close()
        targets = (("dangling", b"gone-\xfe", "\\udcfe"), ("plain", b"file-\xfd", "\\udcfd"))
        for prefix, name, shown in targets:
            for op, link, message in self.receiver_ops(prefix):
                with self.subTest(target=prefix, op=op):
                    os.symlink(os.path.join(os.fsencode(self.session), name),
                               os.fsencode(link))
                    sent = self.assert_sent_as(self.dispatch(self.daemon(), message),
                                               protocol.ERR_NOT_FOUND)
                    self.assertIn(shown, sent["error"])
        self.assertEqual(self.started, [])


class RefusalProseTestCase(unittest.TestCase):
    """The prose rule itself, on both channels a refusal can take."""

    def test_prose_quoting_a_non_utf8_name_encodes_on_both_channels(self) -> None:
        name = "/session/voice/\udcff"
        for built in (protocol.reply_error(f"{name} is not there", protocol.ERR_NOT_FOUND),
                      protocol.dictation_error(f"{name} is not there",
                                               protocol.ERR_NOT_FOUND, segment="listen-1")):
            with self.subTest(keys=sorted(built)):
                frame = protocol.encode(built, limit=protocol.MAX_REPLY_BYTES)
                self.assertEqual(protocol.decode(frame)["error"],
                                 "/session/voice/\\udcff is not there")

    def test_escaped_prose_is_still_cut_to_the_limit(self) -> None:
        limit = protocol.MAX_ERROR_PROSE_CHARS
        reply = protocol.reply_error("\udcff" * limit, protocol.ERR_NOT_FOUND)
        self.assertLessEqual(len(reply["error"]), limit + len(protocol._TRUNCATED))
        self.assertTrue(reply["error"].startswith("\\udcff"))


class OwnControlSocketLiveTestCase(LiveDaemonTestCase):
    """The same refusal from a real kilix-voiced on its real control socket."""

    SETTINGS = ("KILIX_VOICE_TTS_ENGINE=off\n"
                "KILIX_VOICE_STT_ENGINE=vosk\n"
                "KILIX_VOICE_STT_MODEL=small-en-us\n")

    def test_dictate_naming_the_control_socket_is_refused(self) -> None:
        reply = self.request({"op": "dictate", "sock": self.control})
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], protocol.ERR_MALFORMED, reply)
        self.assertFalse(self.request({"op": "status"})["status"]["listening"])
        self.assert_still_serving()

    def test_a_hard_link_to_the_control_socket_is_refused(self) -> None:
        link = os.path.join(self.session_dir, "dictate-7.sock")
        os.link(self.control, link)
        reply = self.request({"op": "dictate", "sock": link})
        self.assertEqual(reply.get("code"), protocol.ERR_MALFORMED, reply)
        self.assert_still_serving()

    def test_chunk_sock_naming_the_control_socket_is_refused(self) -> None:
        reply = self.request({"op": "speak", "text": "Hello there.",
                              "chunk_sock": self.control})
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], protocol.ERR_MALFORMED, reply)
        self.assertFalse(self.request({"op": "status"})["status"]["speaking"])
        self.assert_still_serving()

    def test_non_utf8_names_keep_their_refusal_code_over_the_wire(self) -> None:
        session = os.fsencode(self.session_dir)
        hidden = os.path.join(session, b"\xff")
        os.link(os.fsencode(self.control), hidden)
        cases = (
            ("dictate-8.sock", hidden, {"op": "dictate"}, "sock", protocol.ERR_MALFORMED),
            ("chunks-8.sock", hidden, {"op": "speak", "text": "Hello there."},
             "chunk_sock", protocol.ERR_MALFORMED),
            ("dictate-9.sock", os.path.join(session, b"gone-\xfe"), {"op": "dictate"},
             "sock", protocol.ERR_NOT_FOUND),
        )
        for name, target, message, field, code in cases:
            with self.subTest(name=name):
                link = os.path.join(self.session_dir, name)
                os.symlink(target, os.fsencode(link))
                reply = self.request(dict(message, **{field: link}))
                self.assertIs(reply["ok"], False, reply)
                self.assertEqual(reply["code"], code, reply)
        status = self.request({"op": "status"})["status"]
        self.assertFalse(status["listening"] or status["speaking"], status)
        self.assert_still_serving()


if __name__ == "__main__":
    unittest.main()
