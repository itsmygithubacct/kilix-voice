"""Offline tests for voicelib.protocol.

No socket is bound and nothing is sent anywhere: the wire format is exercised
as bytes, and the containment rules are exercised against real directories and
real symlinks under a temporary root.  ``validate_request`` is the point where
caller-supplied data becomes something the daemon acts on, so its 'sock' check
is tested as the security boundary it is rather than as input tidying.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import tracemalloc
import unittest

from voicelib import protocol


class EncodeTestCase(unittest.TestCase):

    def test_one_utf8_line_with_a_single_trailing_newline(self) -> None:
        line = protocol.encode({"op": "speak", "text": "hello"})
        self.assertIsInstance(line, bytes)
        self.assertTrue(line.endswith(b"\n"))
        self.assertEqual(line.count(b"\n"), 1)
        self.assertEqual(json.loads(line.decode("utf-8")),
                         {"op": "speak", "text": "hello"})

    def test_embedded_newlines_stay_on_one_line(self) -> None:
        # The framing is one message per line, so text containing newlines must
        # be escaped by the encoder rather than splitting the message in two.
        line = protocol.encode({"op": "speak", "text": "a\nb\r\nc"})
        self.assertEqual(line.count(b"\n"), 1)
        self.assertEqual(protocol.decode(line)["text"], "a\nb\r\nc")

    def test_non_ascii_is_written_literally(self) -> None:
        line = protocol.encode({"final": "café ☕"})
        self.assertIn("café ☕", line.decode("utf-8"))

    def test_non_dict_is_refused(self) -> None:
        for msg in ([], "op", None, 3, ("op",)):
            with self.subTest(msg=msg):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.encode(msg)
                self.assertIn("dict", str(caught.exception))

    def test_unserialisable_value_is_refused(self) -> None:
        for msg in ({"a": object()}, {"a": {1, 2}}, {"a": b"bytes"}):
            with self.subTest(msg=list(msg)):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.encode(msg)
                self.assertIn("JSON", str(caught.exception))


class DecodeTestCase(unittest.TestCase):

    def test_round_trip(self) -> None:
        messages = [
            {},
            {"op": "status"},
            {"op": "speak", "text": "hello", "id": "7"},
            {"ok": True, "id": "", "speaking": False, "level": 0.25},
            {"partial": "the quick brown"},
            {"nested": {"a": [1, 2, {"b": None}]}},
            {"text": "ünïcode ✓ \"quoted\" \\ backslash"},
        ]
        for msg in messages:
            with self.subTest(msg=msg):
                self.assertEqual(protocol.decode(protocol.encode(msg)), msg)

    def test_accepts_str_bytes_and_bytearray(self) -> None:
        line = protocol.encode({"op": "status"})
        for raw in (line, bytearray(line), line.decode("utf-8")):
            with self.subTest(kind=type(raw).__name__):
                self.assertEqual(protocol.decode(raw), {"op": "status"})

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        self.assertEqual(protocol.decode(b'  {"op":"status"}  \r\n'),
                         {"op": "status"})

    def test_empty_message_is_refused(self) -> None:
        for raw in (b"", b"\n", b"   \r\n", ""):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.decode(raw)
                self.assertIn("empty", str(caught.exception))

    def test_malformed_json_is_refused(self) -> None:
        for raw in (b"not json", b"{", b'{"op":}', b'{"op":"status",}',
                    b"{'op':'status'}", b'{"op":"status"}{"op":"status"}'):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.decode(raw)
                self.assertIn("JSON", str(caught.exception))

    def test_non_object_json_is_refused(self) -> None:
        for raw in (b"[1,2]", b'"text"', b"3", b"true", b"null"):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.decode(raw)
                self.assertIn("object", str(caught.exception))

    def test_invalid_utf8_is_refused(self) -> None:
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.decode(b'{"op":"speak","text":"\xff\xfe"}')
        self.assertIn("UTF-8", str(caught.exception))

    def test_undecodable_type_is_refused(self) -> None:
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode(None)

    def test_protocol_error_is_a_value_error(self) -> None:
        # Callers already guarding json.loads with ValueError keep working.
        self.assertTrue(issubclass(protocol.ProtocolError, ValueError))


class SessionTestCase(unittest.TestCase):
    """Base fixture: a session directory and a foreign directory beside it."""

    def setUp(self) -> None:
        self.root = os.path.realpath(
            tempfile.mkdtemp(prefix="kilix-voice-proto-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.session = os.path.join(self.root, "session", "voice")
        os.makedirs(self.session, mode=0o700)
        self.outside = os.path.join(self.root, "outside")
        os.makedirs(self.outside, mode=0o700)

    def sock(self, *parts: str) -> str:
        return os.path.join(self.session, *parts)

    def dictate(self, sock: object) -> dict:
        return protocol.validate_request({"op": "dictate", "sock": sock},
                                         self.session)


class SocketContainmentTestCase(SessionTestCase):
    """The 'sock' path is attacker-influenced; containment is the perimeter."""

    def test_accepts_a_pane_socket(self) -> None:
        target = self.sock("dictate-3.sock")
        self.assertEqual(self.dictate(target)["sock"], target)

    def test_accepts_a_socket_that_does_not_exist_yet(self) -> None:
        # The kitty fork may not have bound it at the moment the request lands.
        target = self.sock("dictate-42.sock")
        self.assertFalse(os.path.exists(target))
        self.assertEqual(self.dictate(target)["sock"], target)

    def test_normalises_redundant_syntax(self) -> None:
        expected = self.sock("dictate-3.sock")
        for raw in (self.sock("sub", "..", "dictate-3.sock"),
                    self.sock(".", "dictate-3.sock"),
                    self.session + "//dictate-3.sock",
                    self.sock("a", "b", "..", "..", "dictate-3.sock")):
            with self.subTest(raw=raw):
                self.assertEqual(self.dictate(raw)["sock"], expected)

    def test_accepts_a_nested_path_inside_the_session(self) -> None:
        target = self.sock("panes", "dictate-3.sock")
        self.assertEqual(self.dictate(target)["sock"], target)

    def test_accepts_a_symlink_that_stays_inside_the_session(self) -> None:
        real = self.sock("dictate-3.sock")
        link = self.sock("alias.sock")
        os.symlink(real, link)
        # The resolved path is returned so the daemon connects to what was
        # approved instead of re-resolving the caller's string later.
        self.assertEqual(self.dictate(link)["sock"], real)

    def test_accepts_a_session_dir_reached_through_a_symlink(self) -> None:
        link = os.path.join(self.root, "session-link")
        os.symlink(self.session, link)
        target = self.sock("dictate-3.sock")
        request = protocol.validate_request({"op": "dictate", "sock": target},
                                            link)
        self.assertEqual(request["sock"], target)

    def test_rejects_an_absolute_path_outside_the_session(self) -> None:
        for raw in (os.path.join(self.outside, "evil.sock"),
                    "/tmp/evil.sock",
                    "/etc/passwd",
                    "/dev/log",
                    os.path.dirname(self.session)):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self.dictate(raw)
                self.assertIn("session directory", str(caught.exception))

    def test_rejects_dot_dot_traversal(self) -> None:
        for raw in (self.sock("..", "evil.sock"),
                    self.sock("..", "..", "outside", "evil.sock"),
                    self.sock("a", "..", "..", "evil.sock"),
                    self.session + "/../../../../etc/passwd"):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self.dictate(raw)
                self.assertIn("session directory", str(caught.exception))

    def test_rejects_a_symlink_pointing_outside_the_session(self) -> None:
        victim = os.path.join(self.outside, "evil.sock")
        link = self.sock("dictate-9.sock")
        os.symlink(victim, link)
        with self.assertRaises(protocol.ProtocolError) as caught:
            self.dictate(link)
        message = str(caught.exception)
        self.assertIn("session directory", message)
        self.assertIn(victim, message)

    def test_rejects_a_path_under_a_symlinked_directory(self) -> None:
        # The link is the last component the daemon would have followed, and a
        # lexical check (normpath, startswith) would have let this through.
        os.symlink(self.outside, self.sock("panes"))
        with self.assertRaises(protocol.ProtocolError):
            self.dictate(self.sock("panes", "dictate-3.sock"))

    def test_rejects_a_symlink_to_the_session_parent(self) -> None:
        os.symlink(os.path.dirname(self.session), self.sock("up"))
        with self.assertRaises(protocol.ProtocolError):
            self.dictate(self.sock("up", "evil.sock"))

    def test_rejects_a_sibling_directory_sharing_the_prefix(self) -> None:
        sibling = self.session + "-evil"
        os.makedirs(sibling, mode=0o700)
        with self.assertRaises(protocol.ProtocolError):
            self.dictate(os.path.join(sibling, "dictate-3.sock"))

    def test_rejects_the_session_directory_itself(self) -> None:
        for raw in (self.session, self.session + "/", self.session + "/."):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError):
                    self.dictate(raw)

    def test_rejects_a_relative_path(self) -> None:
        for raw in ("dictate-3.sock", "./dictate-3.sock",
                    "session/voice/dictate-3.sock", "~/dictate-3.sock"):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self.dictate(raw)
                self.assertIn("absolute", str(caught.exception))

    def test_rejects_a_nul_byte(self) -> None:
        with self.assertRaises(protocol.ProtocolError) as caught:
            self.dictate(self.sock("dictate-3.sock\x00/../../evil"))
        self.assertIn("NUL", str(caught.exception))

    def test_rejects_a_missing_or_non_string_sock(self) -> None:
        for raw in (None, "", 3, True, [], {"path": "x"}):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self.dictate(raw)
                self.assertIn("sock", str(caught.exception))

    def test_rejects_dictate_without_a_sock_field(self) -> None:
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_request({"op": "dictate"}, self.session)

    def test_every_accepted_path_resolves_inside_the_session(self) -> None:
        # A belt-and-braces invariant over the accepting cases above: whatever
        # is returned is a real path under the session directory.
        os.symlink(self.sock("dictate-3.sock"), self.sock("alias.sock"))
        for raw in (self.sock("dictate-3.sock"), self.sock("alias.sock"),
                    self.sock("panes", "dictate-1.sock"),
                    self.sock("sub", "..", "dictate-2.sock")):
            with self.subTest(raw=raw):
                resolved = self.dictate(raw)["sock"]
                self.assertTrue(os.path.isabs(resolved))
                self.assertEqual(os.path.commonpath((self.session, resolved)),
                                 self.session)
                self.assertNotEqual(resolved, self.session)


class ValidateRequestTestCase(SessionTestCase):

    def test_every_documented_op_is_accepted(self) -> None:
        self.assertEqual(
            protocol.OPS,
            ("speak", "stop-speech", "dictate", "stop-dictation", "status",
             "ingest-audio"))
        for op in ("stop-speech", "stop-dictation", "status"):
            with self.subTest(op=op):
                self.assertEqual(protocol.validate_request({"op": op},
                                                           self.session),
                                 {"op": op, "id": ""})

    def test_unknown_op_is_refused(self) -> None:
        for msg in ({}, {"op": "shutdown"}, {"op": ""}, {"op": None},
                    {"op": 3}, {"op": ["speak"]}, {"op": "SPEAK"}):
            with self.subTest(msg=msg):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request(msg, self.session)
                self.assertIn("op", str(caught.exception))

    def test_non_object_request_is_refused(self) -> None:
        for msg in ([], "status", None, 7):
            with self.subTest(msg=msg):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.validate_request(msg, self.session)

    def test_speak_keeps_its_text(self) -> None:
        request = protocol.validate_request(
            {"op": "speak", "text": " hello\nworld "}, self.session)
        self.assertEqual(request, {"op": "speak", "id": "",
                                   "text": " hello\nworld "})

    def test_speak_keeps_valid_model_voice_and_rate_overrides(self) -> None:
        request = protocol.validate_request({
            "op": "speak", "text": "hello", "model": "mbrola",
            "voice": "us1", "rate": 200,
        }, self.session)
        self.assertEqual(request, {
            "op": "speak", "id": "", "text": "hello",
            "model": "mbrola", "voice": "us1", "rate": 200,
        })

    def test_speak_refuses_unregistered_models(self) -> None:
        for model in (None, "", "qwen", "../../model", ["espeak"]):
            with self.subTest(model=model):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request({
                        "op": "speak", "text": "hello", "model": model,
                    }, self.session)
                self.assertIn("registered local synthesizer",
                              str(caught.exception))

    def test_speak_refuses_unsafe_voice_tokens(self) -> None:
        for voice in (None, "", "en us", "../../voice", "en-us;id",
                      "x" * 33, ["en-us"]):
            with self.subTest(voice=voice):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request({
                        "op": "speak", "text": "hello", "voice": voice,
                    }, self.session)
                self.assertIn("voice", str(caught.exception))

    def test_speak_refuses_rates_outside_the_shared_presets(self) -> None:
        for rate in (None, True, 0, 119, 171, 999, "170", 170.0):
            with self.subTest(rate=rate):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request({
                        "op": "speak", "text": "hello", "rate": rate,
                    }, self.session)
                self.assertIn("words per minute", str(caught.exception))

    def test_speak_requires_non_empty_text(self) -> None:
        for text in (None, "", "   ", "\n\t", 3, ["hello"], True):
            with self.subTest(text=text):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request({"op": "speak", "text": text},
                                              self.session)
                self.assertIn("text", str(caught.exception))

    def test_ids_are_echoed_as_strings(self) -> None:
        for raw, expected in ((None, ""), ("abc", "abc"), (7, "7"),
                              ("x" * protocol.MAX_ID_CHARS,
                               "x" * protocol.MAX_ID_CHARS)):
            with self.subTest(raw=raw):
                request = protocol.validate_request({"op": "status", "id": raw},
                                                    self.session)
                self.assertEqual(request["id"], expected)

    def test_unusable_ids_are_refused(self) -> None:
        for raw in (True, 1.5, ["a"], {"a": 1}, "x" * (protocol.MAX_ID_CHARS + 1)):
            with self.subTest(raw=raw):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request({"op": "status", "id": raw},
                                              self.session)
                self.assertIn("id", str(caught.exception))

    def test_fields_an_op_does_not_use_are_dropped(self) -> None:
        # Nothing a caller invents may reach dispatch as a surprise keyword —
        # including a 'sock' smuggled in on an op that never validates one.
        request = protocol.validate_request(
            {"op": "speak", "text": "hi", "sock": "/etc/evil.sock",
             "cmd": "rm -rf /", "pitch": 999}, self.session)
        self.assertEqual(set(request), {"op", "id", "text"})

        request = protocol.validate_request(
            {"op": "dictate", "sock": self.sock("dictate-3.sock"),
             "text": "ignored", "submit": "always"}, self.session)
        self.assertEqual(set(request), {"op", "id", "sock"})

        request = protocol.validate_request(
            {"op": "status", "sock": "/etc/evil.sock"}, self.session)
        self.assertEqual(set(request), {"op", "id"})

    def test_the_caller_message_is_not_mutated(self) -> None:
        msg = {"op": "dictate", "sock": self.sock("sub", "..", "dictate-3.sock"),
               "extra": 1}
        before = dict(msg)
        protocol.validate_request(msg, self.session)
        self.assertEqual(msg, before)


class StrDecodeAllocationTestCase(unittest.TestCase):
    """decode(str) refuses an over-size string without a frame-sized copy."""

    PEAK_BOUND = 64 * 1024          # against a 192 KiB input: 3x separation

    def refusal_peak(self, text: str) -> int:
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            baseline = tracemalloc.get_traced_memory()[0]
            with self.assertRaises(protocol.MessageTooLarge):
                protocol.decode(text)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return peak - baseline

    def test_an_over_size_ascii_string_is_refused_without_a_copy(self) -> None:
        text = "x" * (protocol.MAX_REQUEST_BYTES + 1)
        self.assertLess(self.refusal_peak(text), self.PEAK_BOUND)

    def test_an_over_size_multibyte_string_is_refused_without_a_copy(self) -> None:
        text = "€" * (protocol.MAX_REQUEST_BYTES // 3 + 1)
        self.assertLessEqual(len(text), protocol.MAX_REQUEST_BYTES)  # not the char shortcut
        self.assertLess(self.refusal_peak(text), self.PEAK_BOUND)

    def test_a_lone_surrogate_in_a_str_frame_is_a_protocol_error(self) -> None:
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode('{"op":"\ud800"}')

    def test_a_multibyte_string_under_the_limit_still_decodes(self) -> None:  # control
        self.assertEqual(protocol.decode('{"op":"status","id":"café"}')["id"],
                         "café")


class HostileRequestTestCase(SessionTestCase):
    """Caller input raises only ProtocolError, and every reply fits a client."""

    def test_nesting_past_the_recursion_limit_is_a_protocol_error(self) -> None:
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode(b"[" * 100_000)

    def test_an_integer_past_the_digit_limit_is_a_protocol_error(self) -> None:
        for field in ("deadline_ms", "id"):
            with self.subTest(field=field):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.decode(b'{"op":"status","%s":%s}'
                                    % (field.encode(), b"9" * 5000))

    def test_a_lone_surrogate_in_a_socket_path_is_a_protocol_error(self) -> None:
        for field, op in (("sock", "dictate"), ("chunk_sock", "speak")):
            with self.subTest(field=field):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request(
                        {"op": op, "text": "hi", field: self.sock("a\ud800.sock")},
                        self.session)
                self.assertIn("UTF-8", str(caught.exception))

    def test_a_socket_path_past_path_max_is_refused(self) -> None:
        with self.assertRaises(protocol.ProtocolError) as caught:
            self.dictate(self.sock("d" * 5000 + ".sock"))
        self.assertIn("PATH_MAX", str(caught.exception))

    def test_a_refusal_never_quotes_a_value_at_full_length(self) -> None:
        for msg in ({"op": "\\" * 95_000}, {"op": "status", "v": "\\" * 95_000},
                    {"op": "dictate", "sock": '"' * 94_000}):
            with self.subTest(field=next(iter(set(msg) - {"op"}), "op")):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.validate_request(msg, self.session)
                self.assertLess(len(str(caught.exception)), 1000)

    def test_an_error_reply_with_huge_prose_keeps_its_code_and_fits(self) -> None:
        for filler in ("x", "\\", "\x01", "é"):
            with self.subTest(filler=filler):
                reply = {"ok": False, "error": filler * 300_000, "code": "busy"}
                frame = protocol.encode_reply(reply)
                self.assertLessEqual(len(frame), protocol.MAX_REPLY_BYTES)
                self.assertEqual(protocol.decode(frame)["code"], "busy")

    def test_an_over_size_success_reply_becomes_internal_not_partial(self) -> None:
        frame = protocol.encode_reply(protocol.reply_ok("", status="y" * 300_000))
        self.assertLessEqual(len(frame), protocol.MAX_REPLY_BYTES)
        reply = protocol.decode(frame)
        self.assertIs(reply["ok"], False)
        self.assertEqual(reply["code"], protocol.ERR_INTERNAL)

    def test_an_unserialisable_reply_becomes_internal(self) -> None:
        reply = protocol.decode(protocol.encode_reply({"ok": True, "x": object()}))
        self.assertEqual(reply["code"], protocol.ERR_INTERNAL)

    def test_message_too_large_carries_its_size_and_limit(self) -> None:
        with self.assertRaises(protocol.MessageTooLarge) as caught:
            protocol.encode({"text": "x" * 70_000}, limit=protocol.MAX_REPLY_BYTES)
        self.assertEqual(caught.exception.limit, protocol.MAX_REPLY_BYTES)
        self.assertGreater(caught.exception.size, protocol.MAX_REPLY_BYTES)


class ReplyTestCase(unittest.TestCase):

    def test_ok_reply_carries_the_request_id_and_fields(self) -> None:
        reply = protocol.reply_ok("7", speaking=True, listening=False)
        self.assertEqual(reply, {"ok": True, "id": "7", "speaking": True,
                                 "listening": False})
        self.assertEqual(protocol.decode(protocol.encode(reply)), reply)

    def test_ok_reply_defaults_to_an_empty_id(self) -> None:
        self.assertEqual(protocol.reply_ok(), {"ok": True, "id": ""})

    def test_error_reply(self) -> None:
        reply = protocol.reply_error("espeak-ng is not installed. Run: "
                                     "apt install espeak-ng")
        self.assertIs(reply["ok"], False)
        self.assertIn("apt install", reply["error"])
        self.assertEqual(protocol.decode(protocol.encode(reply)), reply)

    def test_error_fields_cannot_replace_what_makes_it_a_refusal(self) -> None:
        for clash in ({"ok": True}, {"error": "fine"}):
            with self.subTest(clash=clash):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.reply_error("x", "busy", **clash)
        reply = protocol.reply_error("x", "busy", protocol={"major": 1})
        self.assertEqual(reply, {"ok": False, "error": "x", "code": "busy",
                                 "protocol": {"major": 1}})

    def test_error_prose_is_capped_at_construction(self) -> None:
        limit = protocol.MAX_ERROR_PROSE_CHARS + len(" …[truncated]")
        self.assertLessEqual(len(protocol.reply_error("x" * 10_000)["error"]), limit)
        self.assertLessEqual(
            len(protocol.dictation_error("x" * 10_000, "busy")["error"]), limit)
        short = "espeak-ng is not installed."
        self.assertEqual(protocol.reply_error(short)["error"], short)

    def test_dictation_datagrams_round_trip(self) -> None:
        for datagram, key in ((protocol.dictation_partial("the quick"), "partial"),
                              (protocol.dictation_final("the quick brown"), "final"),
                              (protocol.dictation_error("no model"), "error")):
            with self.subTest(key=key):
                self.assertEqual(set(datagram), {key})
                self.assertEqual(protocol.decode(protocol.encode(datagram)),
                                 datagram)


if __name__ == "__main__":
    unittest.main()
