"""F104 P1 mechanisms: frame-limit enforcement, caller deadline, error codes.

These three were declared-but-unenforced or wholly absent, which is why P1
vectors V10 (response half), V13 and V16 could not be executed. Each test names
the vector it unblocks.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from voicelib import models, protocol


class FrameLimitTestCase(unittest.TestCase):
    """V10 -- the limit the module declares is now enforced by the module."""

    def test_encode_refuses_a_frame_over_the_declared_limit(self) -> None:
        msg = {"op": "speak", "text": "x" * (protocol.MAX_REQUEST_BYTES + 10)}
        with self.assertRaises(protocol.MessageTooLarge) as caught:
            protocol.encode(msg)
        self.assertIn(f"the limit is {protocol.MAX_REQUEST_BYTES}", str(caught.exception))

    def test_encode_accepts_a_frame_at_the_limit(self) -> None:
        # Positive control: the refusal must be about size, not about the shape.
        text = "x" * (protocol.MAX_REQUEST_BYTES - 64)
        frame = protocol.encode({"op": "speak", "text": text})
        self.assertLessEqual(len(frame), protocol.MAX_REQUEST_BYTES)

    def test_decode_refuses_before_parsing(self) -> None:
        # Not valid JSON either: if the size check did not fire FIRST the error
        # would be a JSON diagnosis, so the message proves the ordering.
        raw = b"{" + b"x" * (protocol.MAX_REQUEST_BYTES + 10)
        with self.assertRaises(protocol.MessageTooLarge) as caught:
            protocol.decode(raw)
        self.assertIn("Refused before decoding", str(caught.exception))

    def test_message_too_large_is_a_protocol_error(self) -> None:
        self.assertTrue(issubclass(protocol.MessageTooLarge, protocol.ProtocolError))

    def test_undecodable_type_still_names_the_type(self) -> None:
        # The size guard must not swallow the existing type diagnosis.
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode(None)


class DeadlineTestCase(unittest.TestCase):
    """V13 -- a caller may bound its wait, and a passed deadline is refused."""

    def setUp(self) -> None:
        self.session = tempfile.mkdtemp(prefix="f104-deadline-")
        os.chmod(self.session, 0o700)

    def _speak(self, **extra):
        msg = {"op": "speak", "text": "hi"}
        msg.update(extra)
        return protocol.validate_request(msg, self.session)

    def test_a_positive_deadline_is_accepted_and_normalised(self) -> None:
        self.assertEqual(self._speak(deadline_ms=5000)["deadline_ms"], 5000)

    def test_absent_deadline_stays_absent(self) -> None:
        self.assertNotIn("deadline_ms", self._speak())

    def test_expired_and_zero_deadlines_are_refused(self) -> None:
        for value in (0, -1, -5000):
            with self.subTest(value=value):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self._speak(deadline_ms=value)
                self.assertIn("must be greater than zero", str(caught.exception))

    def test_untyped_deadlines_are_refused(self) -> None:
        for value in ("5000", 5000.0, True, None, [5000]):
            with self.subTest(value=value):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self._speak(deadline_ms=value)
                self.assertIn("integer number of milliseconds", str(caught.exception))

    def test_absurd_deadline_is_refused(self) -> None:
        with self.assertRaises(protocol.ProtocolError) as caught:
            self._speak(deadline_ms=protocol.MAX_DEADLINE_MS + 1)
        self.assertIn("the ceiling is", str(caught.exception))


class ErrorCodeTestCase(unittest.TestCase):
    """V16 -- errors cross as a code from a closed set, not as prose alone."""

    def test_reply_carries_a_code_from_the_closed_set(self) -> None:
        reply = protocol.reply_error("espeak-ng is not installed.", protocol.ERR_UNAVAILABLE)
        self.assertIs(reply["ok"], False)
        self.assertEqual(reply["code"], protocol.ERR_UNAVAILABLE)
        self.assertIn(reply["code"], protocol.ERROR_CODES)

    def test_default_code_is_internal(self) -> None:
        self.assertEqual(protocol.reply_error("boom")["code"], protocol.ERR_INTERNAL)

    def test_an_unknown_code_is_refused_not_forwarded(self) -> None:
        # The vocabulary must not drift open one caller at a time.
        for bogus in ("oops", "", None, "INTERNAL", 1):
            with self.subTest(code=bogus):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.reply_error("boom", bogus)
                self.assertIn("unknown error code", str(caught.exception))

    def test_every_code_round_trips_on_the_wire(self) -> None:
        for code in protocol.ERROR_CODES:
            with self.subTest(code=code):
                reply = protocol.reply_error("m", code)
                self.assertEqual(protocol.decode(protocol.encode(reply)), reply)

    def test_codes_are_unique_and_lowercase_tokens(self) -> None:
        self.assertEqual(len(set(protocol.ERROR_CODES)), len(protocol.ERROR_CODES))
        for code in protocol.ERROR_CODES:
            self.assertRegex(code, r"^[a-z][a-z-]*[a-z]$")


if __name__ == "__main__":
    unittest.main()


class ProtocolVersionTestCase(unittest.TestCase):
    """V08 -- compatible framing accepts, an incompatible major refuses."""

    def setUp(self) -> None:
        self.session = tempfile.mkdtemp(prefix="f104-version-")
        os.chmod(self.session, 0o700)

    def _speak(self, **extra):
        msg = {"op": "speak", "text": "hi"}
        msg.update(extra)
        return protocol.validate_request(msg, self.session)

    def test_identity_constants_agree(self) -> None:
        self.assertEqual(protocol.PROTOCOL_SCHEMA, "kilix.voice.protocol/v1")
        self.assertEqual(protocol.PROTOCOL_VERSION,
                         f"{protocol.PROTOCOL_MAJOR}.{protocol.PROTOCOL_MINOR}")
        # The schema string must name the same major it claims.
        self.assertTrue(protocol.PROTOCOL_SCHEMA.endswith(f"/v{protocol.PROTOCOL_MAJOR}"))

    def test_accept_arm_same_major_any_minor(self) -> None:
        for value in ("1", "1.0", "1.1", "1.999", 1):
            with self.subTest(value=value):
                self.assertEqual(self._speak(v=value)["v"], str(value))

    def test_refuse_arm_different_major(self) -> None:
        for value in ("2", "2.0", "0.9", 2, "17.3"):
            with self.subTest(value=value):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    self._speak(v=value)
                self.assertIn("is not supported", str(caught.exception))

    def test_omitting_the_version_still_works(self) -> None:
        # Every existing client omits it; that must keep working.
        self.assertNotIn("v", self._speak())

    def test_malformed_version_tokens_are_refused(self) -> None:
        for value in ("", "v1", "1.2.3", "one", "1.", ".1", "-1", 1.0, True, None, ["1"]):
            with self.subTest(value=value):
                with self.assertRaises(protocol.ProtocolError):
                    self._speak(v=value)


class CatalogReaderTestCase(unittest.TestCase):
    """V03 -- a compatible unknown catalog field is ignored, not refused."""

    def _doc(self, **over):
        doc = {
            "schema": models.CATALOG_SCHEMA,
            "default_model": "small-en-us",
            "models": [
                {"id": "small-en-us", "engine": "vosk", "installed": True,
                 "runtime_supported": True, "download_bytes": 41205931},
                {"id": "lgraph-en-us", "engine": "vosk", "installed": False,
                 "runtime_supported": True, "download_bytes": 130557655},
            ],
        }
        doc.update(over)
        return doc

    def test_the_real_producer_document_round_trips(self) -> None:
        # The strongest control available: parse what the shipped producer emits.
        out = subprocess.run([sys.executable, "kilix-stt", "--models", "--json"],
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest(f"kilix-stt --models-json unavailable: {out.stderr[:120]}")
        parsed = models.read_catalog(json.loads(out.stdout))
        self.assertEqual(parsed["schema"], models.CATALOG_SCHEMA)
        self.assertTrue(parsed["models"])

    def test_unknown_fields_are_preserved_not_refused(self) -> None:
        doc = self._doc(future_top_level={"anything": 1})
        doc["models"][0]["future_record_field"] = ["whatever"]
        parsed = models.read_catalog(doc)
        self.assertEqual(parsed["future_top_level"], {"anything": 1})
        self.assertEqual(parsed["models"][0]["future_record_field"], ["whatever"])

    def test_a_different_schema_is_refused(self) -> None:
        for schema in ("kilix.speech.models/v2", "other/v1", "", None, 1):
            with self.subTest(schema=schema):
                with self.assertRaises(models.CatalogError) as caught:
                    models.read_catalog(self._doc(schema=schema))
                self.assertIn("this reader speaks", str(caught.exception))

    def test_a_mistyped_known_field_is_refused(self) -> None:
        doc = self._doc()
        doc["models"][0]["installed"] = "yes"
        with self.assertRaises(models.CatalogError) as caught:
            models.read_catalog(doc)
        self.assertIn("must be bool", str(caught.exception))

    def test_a_bool_is_not_accepted_as_an_int(self) -> None:
        doc = self._doc()
        doc["models"][0]["download_bytes"] = True
        with self.assertRaises(models.CatalogError):
            models.read_catalog(doc)

    def test_duplicate_ids_are_refused(self) -> None:
        doc = self._doc()
        doc["models"][1]["id"] = "small-en-us"
        with self.assertRaises(models.CatalogError) as caught:
            models.read_catalog(doc)
        self.assertIn("duplicate model id", str(caught.exception))

    def test_a_default_outside_the_records_is_refused(self) -> None:
        with self.assertRaises(models.CatalogError) as caught:
            models.read_catalog(self._doc(default_model="not-present"))
        self.assertIn("not one of the", str(caught.exception))

    def test_missing_required_fields_are_refused(self) -> None:
        for key in ("id", "engine"):
            with self.subTest(key=key):
                doc = self._doc()
                del doc["models"][0][key]
                with self.assertRaises(models.CatalogError) as caught:
                    models.read_catalog(doc)
                self.assertIn(f"missing required field {key!r}", str(caught.exception))
