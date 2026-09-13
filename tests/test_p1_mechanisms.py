"""F104 P1 mechanisms: frame-limit enforcement, caller deadline, error codes.

These three were declared-but-unenforced or wholly absent, which is why P1
vectors V10 (response half), V13 and V16 could not be executed. Each test names
the vector it unblocks.
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from voicelib import consent, models, protocol


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


class ConsentTestCase(unittest.TestCase):
    """V25 S02/S03 -- consent is created, reused, and invalidated by digest."""

    A = "a" * 64
    B = "b" * 64

    def setUp(self) -> None:
        self.home = os.path.realpath(tempfile.mkdtemp(prefix="f104-consent-"))
        self.env = mock.patch.dict(os.environ, {"HOME": self.home}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(shutil.rmtree, self.home, True)

    def test_import_touches_no_filesystem(self) -> None:
        self.assertFalse(os.path.exists(os.path.join(self.home, ".local")))

    def test_absent_record_is_not_consent(self) -> None:
        self.assertFalse(consent.granted("dictation", self.A))

    def test_grant_then_reuse(self) -> None:                      # S02
        consent.grant("dictation", self.A)
        self.assertTrue(consent.granted("dictation", self.A))
        self.assertTrue(consent.granted("dictation", self.A))     # reuse, not re-grant

    def test_grant_is_idempotent(self) -> None:                   # S02
        first = consent.grant("dictation", self.A)
        second = consent.grant("dictation", self.A)
        self.assertEqual(first, second)

    def test_a_changed_digest_invalidates(self) -> None:          # S03
        consent.grant("dictation", self.A)
        self.assertFalse(consent.granted("dictation", self.B))
        # and it is not an error -- the caller simply asks again
        consent.grant("dictation", self.B)
        self.assertTrue(consent.granted("dictation", self.B))
        self.assertFalse(consent.granted("dictation", self.A))

    def test_subjects_are_independent(self) -> None:
        consent.grant("dictation", self.A)
        self.assertFalse(consent.granted("synthesis", self.A))

    def test_the_record_is_private_and_inside_the_private_layout(self) -> None:
        consent.grant("dictation", self.A)
        path = consent.consent_path()
        self.assertTrue(path.startswith(self.home + "/.local/gpu_terminal/"))
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode), 0o700)

    def test_mode_is_not_umask_dependent(self) -> None:
        # Real property, but note what enforces it: tempfile.mkstemp creates
        # 0600 whatever the umask. Deleting the explicit fchmod in _write does
        # NOT turn this red. Recorded so nobody reads a pass here as proof that
        # the fchmod is load-bearing.
        old = os.umask(0o000)
        try:
            consent.grant("dictation", self.A)
            self.assertEqual(stat.S_IMODE(os.stat(consent.consent_path()).st_mode), 0o600)
        finally:
            os.umask(old)

    def test_revoke(self) -> None:
        consent.grant("dictation", self.A)
        self.assertTrue(consent.revoke("dictation"))
        self.assertFalse(consent.granted("dictation", self.A))
        self.assertFalse(consent.revoke("dictation"))

    def test_malformed_subjects_and_digests_are_refused(self) -> None:
        for subject in ("", "a/b", "../x", "-lead", "x" * 65, None, 1, "a b"):
            with self.subTest(subject=subject):
                with self.assertRaises(consent.ConsentError):
                    consent.grant(subject, self.A)
        for digest in ("", "abc", "A" * 64, "g" * 64, "a" * 63, None, 1):
            with self.subTest(digest=digest):
                with self.assertRaises(consent.ConsentError):
                    consent.grant("dictation", digest)

    def test_a_corrupt_record_is_refused_not_treated_as_absent(self) -> None:
        consent.grant("dictation", self.A)
        with open(consent.consent_path(), "w") as handle:
            handle.write("{not json")
        with self.assertRaises(consent.ConsentError) as caught:
            consent.granted("dictation", self.A)
        self.assertIn("not valid JSON", str(caught.exception))

    def test_a_foreign_schema_is_refused(self) -> None:
        consent.grant("dictation", self.A)
        with open(consent.consent_path(), "w") as handle:
            json.dump({"schema": "other/v1", "grants": {"dictation": self.A}}, handle)
        with self.assertRaises(consent.ConsentError):
            consent.granted("dictation", self.A)

    def test_no_temporary_file_is_left_behind(self) -> None:
        consent.grant("dictation", self.A)
        leftovers = [n for n in os.listdir(os.path.dirname(consent.consent_path()))
                     if n.startswith(".consent-")]
        self.assertEqual(leftovers, [])


class AudioLimitTestCase(unittest.TestCase):
    """V21/V22 -- A06 and A07 refuse rather than truncate or embed."""

    def test_over_the_audio_limit_is_refused_not_truncated(self) -> None:
        with self.assertRaises(protocol.MessageTooLarge) as caught:
            protocol.check_audio_bytes(protocol.MAX_AUDIO_BYTES + 1)
        self.assertIn("refused, not truncated", str(caught.exception))

    def test_at_the_limit_is_accepted(self) -> None:          # positive control
        self.assertEqual(protocol.check_audio_bytes(protocol.MAX_AUDIO_BYTES),
                         protocol.MAX_AUDIO_BYTES)

    def test_the_embedded_ceiling_is_far_smaller(self) -> None:
        self.assertLess(protocol.MAX_EMBEDDED_AUDIO_BYTES, protocol.MAX_AUDIO_BYTES)
        size = protocol.MAX_EMBEDDED_AUDIO_BYTES + 1
        with self.assertRaises(protocol.MessageTooLarge) as caught:
            protocol.check_audio_bytes(size, embedded=True)
        self.assertIn("Send a descriptor", str(caught.exception))
        protocol.check_audio_bytes(size)                      # fine unembedded

    def test_negative_and_untyped_lengths_are_refused(self) -> None:
        for value in (-1, "10", 1.0, True, None):
            with self.subTest(value=value):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.check_audio_bytes(value)


class TranscriptMetadataTestCase(unittest.TestCase):
    """V23 -- A09 segment identifiers, A10 word timestamps, A11 speaker."""

    def test_partial_segments_are_unstable_and_final_stable(self) -> None:
        self.assertIs(protocol.dictation_partial("the qu", "s1")["stable"], False)
        self.assertIs(protocol.dictation_final("the quick", "s1")["stable"], True)

    def test_metadata_is_optional_so_nothing_is_fabricated(self) -> None:
        plain = protocol.dictation_final("hi")
        for key in ("segment", "stable", "words", "speaker"):
            self.assertNotIn(key, plain)

    def test_word_timestamps_round_trip(self) -> None:
        words = [{"word": "the", "start_ms": 0, "end_ms": 120},
                 {"word": "quick", "start_ms": 120, "end_ms": 400}]
        out = protocol.dictation_final("the quick", "s1", words=words)
        self.assertEqual(out["words"], words)
        self.assertEqual(protocol.decode(protocol.encode(out)), out)

    def test_backwards_and_inverted_timestamps_are_refused(self) -> None:
        inverted = [{"word": "a", "start_ms": 200, "end_ms": 100}]
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.dictation_final("a", "s1", words=inverted)
        self.assertIn("before it starts", str(caught.exception))
        backwards = [{"word": "a", "start_ms": 0, "end_ms": 200},
                     {"word": "b", "start_ms": 100, "end_ms": 300}]
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.dictation_final("a b", "s1", words=backwards)
        self.assertIn("must not go backwards", str(caught.exception))

    def test_a_speaker_label_without_confidence_is_refused(self) -> None:
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.dictation_final("hi", "s1", speaker={"label": "spk1"})
        self.assertIn("requires a speaker confidence", str(caught.exception))

    def test_speaker_confidence_is_bounded(self) -> None:
        for bad in (-0.1, 1.1, 2, True, "0.5", None):
            with self.subTest(confidence=bad):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.dictation_final("hi", "s1",
                                             speaker={"label": "spk1", "confidence": bad})
        ok = protocol.dictation_final("hi", "s1", speaker={"label": "spk1", "confidence": 0.9})
        self.assertEqual(ok["speaker"], {"label": "spk1", "confidence": 0.9})


class SynthesisChunkTestCase(unittest.TestCase):
    """V24 -- A12 sequence, A13 provenance, A14 seed/settings, A15 bounded."""

    def _chunk(self, **over):
        kw = dict(sequence=0, pcm_bytes=3200, sample_rate=24000,
                  voice="en-us", model="piper-en-us-kristin-medium")
        kw.update(over)
        return protocol.synthesis_chunk(**kw)

    def test_a_chunk_carries_sequence_provenance_and_bounds(self) -> None:
        chunk = self._chunk(seed=7, temperature=0.0)
        self.assertEqual(chunk["sequence"], 0)
        self.assertEqual(chunk["voice"], "en-us")
        self.assertEqual(chunk["model"], "piper-en-us-kristin-medium")
        self.assertEqual(chunk["seed"], 7)
        self.assertEqual(chunk["settings"], {"temperature": 0.0})
        self.assertIs(chunk["final"], False)
        self.assertEqual(protocol.decode(protocol.encode(chunk)), chunk)

    def test_samples_never_ride_in_the_json(self) -> None:      # A07
        chunk = self._chunk()
        self.assertNotIn("pcm", chunk)
        self.assertNotIn("audio", chunk)
        self.assertIsInstance(chunk["pcm_bytes"], int)

    def test_an_oversized_chunk_is_refused(self) -> None:       # A06 via A15
        with self.assertRaises(protocol.MessageTooLarge):
            self._chunk(pcm_bytes=protocol.MAX_AUDIO_BYTES + 1)

    def test_malformed_chunk_fields_are_refused(self) -> None:
        for kw in ({"sequence": -1}, {"sequence": "0"}, {"sample_rate": 0},
                   {"sample_rate": -1}, {"voice": "en us"}, {"model": ""},
                   {"seed": "7"}):
            with self.subTest(**kw):
                with self.assertRaises(protocol.ProtocolError):
                    self._chunk(**kw)

    def test_a_final_chunk_is_marked(self) -> None:             # A15
        self.assertIs(self._chunk(sequence=9, final=True)["final"], True)
