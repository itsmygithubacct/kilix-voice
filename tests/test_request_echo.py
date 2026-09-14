"""A caller's own string is never answered `internal`.

Wave-1 residual: {"op":"status","id":"\\ud800"} passed validation, the reply
echoed the id, the reply could not be encoded, and encode_reply's last resort
answered code `internal` -- "this is a bug" -- for the caller's own input. The
id is the one caller string a reply echoes verbatim; every other string a
request carries is either validated against an ASCII vocabulary, quoted back
through repr(), or never echoed. The sweep below drives each of them over the
real control socket so that claim is an observation, not a reading.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from voicelib import protocol

from tests.livedaemon import LiveDaemonTestCase

SURROGATE = "\ud800"


class RequestIdEncodabilityTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.session = os.path.realpath(tempfile.mkdtemp(prefix="kv-echo-"))
        self.addCleanup(shutil.rmtree, self.session, True)

    def test_an_id_that_is_not_utf8_encodable_is_refused(self) -> None:
        for raw in (SURROGATE, "job\udfff", "\udc80" * protocol.MAX_ID_CHARS):
            with self.subTest(raw=ascii(raw)):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request({"op": "status", "id": raw},
                                              self.session)
                # No code on the instance: the daemon's validation arm then
                # supplies `malformed`, which the socket test below observes.
                self.assertIsNone(caught.exception.code)
                self.assertIn("'id'", str(caught.exception))

    def test_a_non_ascii_id_that_encodes_is_still_echoed(self) -> None:  # control
        request = protocol.validate_request({"op": "status", "id": "café-☕"},
                                            self.session)
        self.assertEqual(request["id"], "café-☕")
        reply = protocol.decode(protocol.encode_reply(protocol.reply_ok(request["id"])))
        self.assertEqual(reply, {"ok": True, "id": "café-☕"})


class CallerStringsOverTheControlSocketTestCase(LiveDaemonTestCase):

    def test_a_surrogate_id_is_malformed_and_the_daemon_serves_on(self) -> None:
        raw = self.exchange(b'{"op":"status","id":"\\ud800"}\n')
        self.assertTrue(0 < len(raw) <= 65535, self.log_tail())
        reply = json.loads(raw.decode("utf-8"))
        self.assertIs(reply["ok"], False, reply)
        self.assertEqual(reply["code"], protocol.ERR_MALFORMED, reply)
        self.assertNotIn("bug", reply["error"])
        self.assert_still_serving()

    def test_no_string_field_holding_a_lone_surrogate_is_answered_internal(self) -> None:
        inside = os.path.join(self.session_dir, f"dictate-{SURROGATE}.sock")
        requests = (
            {"op": SURROGATE},
            {"op": "status" + SURROGATE},
            {"op": "status", "id": SURROGATE},
            {"op": "stop-speech", "id": "x" + SURROGATE},
            {"op": "stop-dictation", "id": SURROGATE},
            {"op": "status", "v": SURROGATE},
            {"op": "status", "v": "1." + SURROGATE},
            {"op": "status", "deadline_ms": SURROGATE},
            {"op": "speak", "text": SURROGATE},
            {"op": "speak", "text": "hi " + SURROGATE},
            {"op": "speak", "text": "hi", "model": SURROGATE},
            {"op": "speak", "text": "hi", "voice": SURROGATE},
            {"op": "speak", "text": "hi", "rate": SURROGATE},
            {"op": "speak", "text": "hi", "chunk_sock": inside},
            {"op": "dictate", "sock": inside},
            {"op": "dictate", "sock": SURROGATE},
        )
        for message in requests:
            with self.subTest(message=ascii(message)):
                reply = self.request(message)
                if reply.get("ok") is not True:
                    self.assertIn(reply.get("code"), protocol.ERROR_CODES, reply)
                    self.assertNotEqual(reply["code"], protocol.ERR_INTERNAL, reply)
                self.assertIsNone(self.daemon.poll(), self.log_tail())
        self.assert_still_serving()


if __name__ == "__main__":
    unittest.main()
