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

from voicelib import consent, models, protocol, resources


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
        # An earlier version of this comment claimed mkstemp gives 0600
        # "whatever the umask" and that the explicit fchmod was not
        # load-bearing. WRONG, and an independent review caught it: a umask can
        # only narrow creation. Measured: umask 0000 -> 0600, 0077 -> 0600,
        # 0777 -> 0000. See test_mode_survives_a_hostile_umask, which is the
        # arm that actually discriminates.
        old = os.umask(0o000)
        try:
            consent.grant("dictation", self.A)
            self.assertEqual(stat.S_IMODE(os.stat(consent.consent_path()).st_mode), 0o600)
        finally:
            os.umask(old)

    def test_mode_survives_a_hostile_umask(self) -> None:
        # THIS is the discriminating arm: under umask 0777 mkstemp yields 0000
        # and only the explicit fchmod restores 0600. Deleting that fchmod
        # turns this red.
        # Establish the directories under a normal umask first. Creating them
        # under 0777 fails outright, because paths.ensure_private_dir hardens
        # only the LEAF: os.makedirs(parent, mode=DIR_MODE) has the umask strip
        # every ancestor bit. That is a real pre-existing defect in a pinned
        # seam, recorded separately; it is not what this test is about.
        consent.grant("dictation", self.A)
        os.unlink(consent.consent_path())
        old = os.umask(0o777)
        try:
            consent.grant("dictation", self.A)
            self.assertEqual(stat.S_IMODE(os.stat(consent.consent_path()).st_mode),
                             0o600)
        finally:
            os.umask(old)

    def test_concurrent_grants_of_different_subjects_are_all_retained(self) -> None:
        # Finding 7: atomic rename gives complete-file visibility, not isolated
        # transactions. This forces the LOST-UPDATE interleaving deterministically
        # instead of hoping two threads race: writer A is held between its read
        # and its write while writer B completes. Without the transaction lock A
        # then writes back a record that never saw B. With the lock, B cannot
        # start until A has finished, so both survive.
        import threading
        a_has_read = threading.Event()
        b_has_finished = threading.Event()
        real_load = consent._load
        first_call = {"done": False}

        def slow_load():
            record = real_load()
            if not first_call["done"]:
                first_call["done"] = True
                a_has_read.set()
                b_has_finished.wait(timeout=5)
            return record

        errors = []

        def writer_a():
            try:
                with mock.patch.object(consent, "_load", slow_load):
                    consent.grant("first", self.A)
            except BaseException as error:
                errors.append(error)

        def writer_b():
            try:
                a_has_read.wait(timeout=5)
                consent.grant("second", self.B)
            finally:
                b_has_finished.set()

        threads = [threading.Thread(target=writer_a), threading.Thread(target=writer_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(errors, [])
        self.assertTrue(consent.granted("first", self.A), "writer A's grant was lost")
        self.assertTrue(consent.granted("second", self.B), "writer B's grant was lost")

    def test_the_transaction_is_actually_exclusive(self) -> None:
        # Direct check of the mechanism, not of a race: while one transaction is
        # open, a second must not be able to enter.
        import threading
        inside = threading.Event()
        release = threading.Event()
        second_entered = threading.Event()

        def holder():
            with consent._transaction():
                inside.set()
                release.wait(timeout=5)

        def contender():
            inside.wait(timeout=5)
            with consent._transaction():
                second_entered.set()

        h = threading.Thread(target=holder); c = threading.Thread(target=contender)
        h.start(); c.start()
        inside.wait(timeout=5)
        self.assertFalse(second_entered.wait(timeout=0.5),
                         "a second transaction entered while the first was open")
        release.set()
        h.join(timeout=5); c.join(timeout=5)
        self.assertTrue(second_entered.is_set(), "the lock was never released")

    def test_a_malformed_grant_entry_is_refused_not_read_as_absent(self) -> None:
        consent.grant("dictation", self.A)
        with open(consent.consent_path(), "w") as handle:
            json.dump({"schema": consent.CONSENT_SCHEMA,
                       "grants": {"dictation": "not-a-digest"}}, handle)
        with self.assertRaises(consent.ConsentError) as caught:
            consent.granted("dictation", self.A)
        self.assertIn("malformed grant", str(caught.exception))

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


class ResourceProfileTestCase(unittest.TestCase):
    """C06/C12/C18 -- a device class and a MEASURED resource profile."""

    def _p(self, **over):
        profile = {
            "schema": resources.RESOURCE_SCHEMA,
            "device_class": resources.DEVICE_CUDA,
            "measured": {"host": "pleon", "date": "2026-09-13",
                         "peak_vram_mib": 3629, "peak_ram_mib": 2048},
        }
        profile.update(over)
        return profile

    def test_a_conforming_profile_validates(self) -> None:
        self.assertEqual(resources.validate(self._p())["device_class"], "cuda")

    def test_a_figure_without_a_host_or_date_is_refused(self) -> None:
        for drop in ("host", "date"):
            with self.subTest(drop=drop):
                bad = self._p()
                del bad["measured"][drop]
                with self.assertRaises(resources.ResourceError):
                    resources.validate(bad)

    def test_an_empty_measurement_is_refused(self) -> None:
        # An empty profile would certify a model as measured when it is not.
        with self.assertRaises(resources.ResourceError) as caught:
            resources.validate(self._p(measured={"host": "pleon", "date": "2026-09-13"}))
        self.assertIn("carries no figure at all", str(caught.exception))

    def test_an_accelerator_must_report_vram(self) -> None:
        bad = self._p()
        del bad["measured"]["peak_vram_mib"]
        with self.assertRaises(resources.ResourceError) as caught:
            resources.validate(bad)
        self.assertIn("must report peak_vram_mib", str(caught.exception))

    def test_a_cpu_profile_reporting_vram_is_a_measurement_error(self) -> None:
        with self.assertRaises(resources.ResourceError) as caught:
            resources.validate(self._p(device_class=resources.DEVICE_CPU))
        self.assertIn("measurement error", str(caught.exception))

    def test_unknown_device_classes_are_refused(self) -> None:
        for bad in ("gpu", "CUDA", "", None, 1):
            with self.subTest(device=bad):
                with self.assertRaises(resources.ResourceError):
                    resources.validate(self._p(device_class=bad))

    def test_absent_headroom_is_unknown_not_infinite(self) -> None:
        profile = self._p()
        self.assertFalse(resources.fits(profile))                       # nothing known
        self.assertFalse(resources.fits(profile, available_vram_mib=8192))  # ram unknown
        self.assertTrue(resources.fits(profile, available_vram_mib=8192,
                                       available_ram_mib=46000))
        self.assertFalse(resources.fits(profile, available_vram_mib=2048,
                                        available_ram_mib=46000))

    def test_a_lone_demand_still_needs_its_own_headroom(self) -> None:
        # Found by independent review: a profile measuring only model_bytes
        # returned True with NO headroom supplied at all, contradicting the
        # documented "absent headroom is unknown".
        cpu_only_disk = {
            "schema": resources.RESOURCE_SCHEMA,
            "device_class": resources.DEVICE_CPU,
            "measured": {"host": "h", "date": "2026-09-13", "model_bytes": 10,
                         "peak_ram_mib": 1},
        }
        self.assertFalse(resources.fits(cpu_only_disk))
        self.assertFalse(resources.fits(cpu_only_disk, available_vram_mib=99))
        self.assertTrue(resources.fits(cpu_only_disk, available_disk_bytes=10,
                                       available_ram_mib=1))
        self.assertFalse(resources.fits(cpu_only_disk, available_disk_bytes=9,
                                        available_ram_mib=1))

    def test_malformed_headroom_is_refused_not_certified(self) -> None:
        # NaN in particular makes every comparison false, which reads as "fits".
        profile = self._p()
        for bad in (float("nan"), float("inf"), "lots", True, -1, 1.5):
            with self.subTest(headroom=bad):
                with self.assertRaises(resources.ResourceError):
                    resources.fits(profile, available_vram_mib=bad,
                                   available_ram_mib=46000)

    def test_pleon_measurements_fit_its_measured_headroom(self) -> None:
        # The figures actually measured on pleon under the H2 fixture.
        self.assertTrue(resources.fits(self._p(), available_vram_mib=8192,
                                       available_ram_mib=46000))


class LegacyCatalogEntryTestCase(unittest.TestCase):
    """V01/C01 -- a legacy entry stays valid though optional fields now exist."""

    def test_a_five_field_spec_still_constructs(self) -> None:
        # This, not an exact _fields tuple, is what "remains valid without new
        # optional fields" means. Asserting the tuple length would forbid C06
        # and C12 from ever being added, which is not what C01 asks.
        spec = models.ModelSpec("legacy-id", models.ENGINE_VOSK, 1, True, "s")
        self.assertEqual(spec.catalog_id, "legacy-id")
        self.assertEqual(spec.device_class, resources.DEVICE_CPU)
        self.assertIsNone(spec.resource_profile)

    def test_the_shipped_legacy_entries_are_untouched(self) -> None:
        for legacy in ("small-en-us", "lgraph-en-us"):
            with self.subTest(legacy=legacy):
                spec = models.MODEL_BY_ID[legacy]
                self.assertEqual(spec.engine, models.ENGINE_VOSK)
                self.assertEqual(spec.device_class, resources.DEVICE_CPU)
                self.assertIsNone(spec.resource_profile)

    def test_a_catalog_record_may_carry_a_profile(self) -> None:
        doc = {"schema": models.CATALOG_SCHEMA, "default_model": "m",
               "models": [{"id": "m", "engine": "vosk",
                           "device_class": "cuda",
                           "resource_profile": {
                               "schema": resources.RESOURCE_SCHEMA,
                               "device_class": "cuda",
                               "measured": {"host": "pleon", "date": "2026-09-13",
                                            "peak_vram_mib": 3629,
                                            "peak_ram_mib": 2048}}}]}
        self.assertEqual(models.read_catalog(doc)["models"][0]["device_class"], "cuda")

    def test_contradictory_device_declarations_are_refused(self) -> None:
        # Finding 6: both were validated individually and never compared, so a
        # cuda model could be scheduled onto a cpu.
        doc = {"schema": models.CATALOG_SCHEMA,
               "models": [{"id": "m", "engine": "vosk", "device_class": "cpu",
                           "resource_profile": {
                               "schema": resources.RESOURCE_SCHEMA,
                               "device_class": "cuda",
                               "measured": {"host": "h", "date": "2026-09-13",
                                            "peak_vram_mib": 100,
                                            "peak_ram_mib": 64}}}]}
        with self.assertRaises(models.CatalogError) as caught:
            models.read_catalog(doc)
        self.assertIn("they must", str(caught.exception))

    def test_agreeing_device_declarations_are_accepted(self) -> None:   # control
        doc = {"schema": models.CATALOG_SCHEMA,
               "models": [{"id": "m", "engine": "vosk", "device_class": "cuda",
                           "resource_profile": {
                               "schema": resources.RESOURCE_SCHEMA,
                               "device_class": "cuda",
                               "measured": {"host": "h", "date": "2026-09-13",
                                            "peak_vram_mib": 100,
                                            "peak_ram_mib": 64}}}]}
        self.assertEqual(models.read_catalog(doc)["models"][0]["device_class"], "cuda")

    def test_a_malformed_profile_is_refused_at_the_catalog_boundary(self) -> None:
        doc = {"schema": models.CATALOG_SCHEMA,
               "models": [{"id": "m", "engine": "vosk",
                           "resource_profile": {"schema": "wrong/v1"}}]}
        with self.assertRaises(models.CatalogError) as caught:
            models.read_catalog(doc)
        self.assertIn("resource_profile is invalid", str(caught.exception))


class R2SurvivorTestCase(unittest.TestCase):
    """The 11 remaining mutations that survived independent review R2.

    Each is an exactness or boundary distinction the earlier tests did not make:
    they proved a refusal happens somewhere past the limit, not that it happens
    at exactly the limit and not one byte before it.
    """

    # --- frame_exact_endpoint / audio_embedded_endpoint -------------------
    def test_frame_limit_endpoint_is_exact(self) -> None:
        pad = protocol.MAX_REQUEST_BYTES - len(protocol.encode({"k": ""})) 
        at = {"k": "x" * pad}
        self.assertEqual(len(protocol.encode(at)), protocol.MAX_REQUEST_BYTES)
        over = {"k": "x" * (pad + 1)}
        with self.assertRaises(protocol.MessageTooLarge):
            protocol.encode(over)

    def test_audio_endpoints_are_exact(self) -> None:
        for embedded, limit in ((False, protocol.MAX_AUDIO_BYTES),
                                (True, protocol.MAX_EMBEDDED_AUDIO_BYTES)):
            with self.subTest(embedded=embedded):
                self.assertEqual(
                    protocol.check_audio_bytes(limit, embedded=embedded), limit)
                with self.assertRaises(protocol.MessageTooLarge):
                    protocol.check_audio_bytes(limit + 1, embedded=embedded)

    # --- synthesis_sample_rate / synthesis_length -------------------------
    def test_synthesis_metadata_is_carried_accurately(self) -> None:
        chunk = protocol.synthesis_chunk(3, pcm_bytes=4096, sample_rate=22050,
                                         voice="en-gb", model="m", seed=11)
        self.assertEqual(chunk["sample_rate"], 22050)   # not a default
        self.assertEqual(chunk["pcm_bytes"], 4096)      # not a placeholder
        self.assertEqual(chunk["sequence"], 3)
        self.assertEqual(chunk["seed"], 11)

    def test_synthesis_rejects_a_zero_or_negative_rate_at_the_boundary(self) -> None:
        protocol.synthesis_chunk(0, pcm_bytes=0, sample_rate=1,
                                 voice="a", model="m")     # 1 Hz is legal
        for rate in (0, -1):
            with self.subTest(rate=rate):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.synthesis_chunk(0, pcm_bytes=0, sample_rate=rate,
                                             voice="a", model="m")

    # --- resource_ram_comparison / resource_measurement_types -------------
    def test_ram_shortfall_is_detected_at_the_boundary(self) -> None:
        profile = {"schema": resources.RESOURCE_SCHEMA,
                   "device_class": resources.DEVICE_CPU,
                   "measured": {"host": "h", "date": "2026-09-13",
                                "peak_ram_mib": 100}}
        self.assertTrue(resources.fits(profile, available_ram_mib=100))
        self.assertFalse(resources.fits(profile, available_ram_mib=99))

    def test_measurement_figures_must_be_non_negative_integers(self) -> None:
        for bad in (-1, 1.5, True, "100", None):
            with self.subTest(value=bad):
                with self.assertRaises(resources.ResourceError):
                    resources.validate({
                        "schema": resources.RESOURCE_SCHEMA,
                        "device_class": resources.DEVICE_CPU,
                        "measured": {"host": "h", "date": "2026-09-13",
                                     "peak_ram_mib": bad}})
        resources.validate({"schema": resources.RESOURCE_SCHEMA,
                            "device_class": resources.DEVICE_CPU,
                            "measured": {"host": "h", "date": "2026-09-13",
                                         "peak_ram_mib": 0}})   # zero is legal

    # --- producer_device_class / producer_resource_profile ----------------
    def _produced(self):
        import subprocess as sp
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = sp.run([sys.executable, "kilix-stt", "--models", "--json"],
                     cwd=root, capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest(out.stderr[:150])
        return json.loads(out.stdout)

    def test_the_producer_publishes_a_device_class_on_every_record(self) -> None:
        doc = self._produced()
        self.assertTrue(doc["models"])
        for record in doc["models"]:
            with self.subTest(model=record["id"]):
                self.assertIn("device_class", record)
                self.assertIn(record["device_class"], resources.DEVICE_CLASSES)

    def test_the_producer_emits_a_measured_profile_when_one_exists(self) -> None:
        # No shipped spec carries one yet, so drive the branch directly rather
        # than asserting on absence -- absence would pass even if the branch
        # were deleted.
        spec = models.ModelSpec(
            "m", models.ENGINE_VOSK, 1, True, "s", resources.DEVICE_CUDA,
            {"schema": resources.RESOURCE_SCHEMA, "device_class": "cuda",
             "measured": {"host": "h", "date": "2026-09-13",
                          "peak_vram_mib": 10, "peak_ram_mib": 8}})
        self.assertIsNotNone(spec.resource_profile)
        source = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kilix-stt")).read()
        self.assertIn('records[-1]["resource_profile"] = spec.resource_profile',
                      source)
        self.assertIn("if spec.resource_profile is not None:", source)


class R2Findings4And6TestCase(unittest.TestCase):
    """R2 finding 4 (dictation errors prose-only) and 6 (actionable fields)."""

    def test_the_bare_dictation_error_shape_is_unchanged(self) -> None:
        # The pinned test at tests/test_protocol.py:419 asserts exactly {"error"}.
        # Defaulting a code in would change a shipped wire shape.
        self.assertEqual(set(protocol.dictation_error("x")), {"error"})

    def test_a_dictation_error_can_carry_a_closed_code(self) -> None:
        out = protocol.dictation_error("x", protocol.ERR_UNAVAILABLE)
        self.assertEqual(out["code"], protocol.ERR_UNAVAILABLE)
        with self.assertRaises(protocol.ProtocolError):
            protocol.dictation_error("x", "bogus")

    def test_the_runtime_dictation_error_supplies_a_code(self) -> None:
        source = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kilix-voiced")).read()
        self.assertIn("protocol.ERR_UNAVAILABLE", source)

    # --- finding 6: install_and_default_argv is action-bearing -------------
    def _doc(self, argv):
        return {"schema": models.CATALOG_SCHEMA,
                "models": [{"id": "m", "engine": "vosk",
                            "install_and_default_argv": argv}]}

    def test_the_producers_own_argv_is_accepted(self) -> None:      # control
        good = ["kilix", "stt", "--install", "small-en-us",
                "--default", "small-en-us"]
        self.assertEqual(
            models.read_catalog(self._doc(good))["models"][0]
            ["install_and_default_argv"], good)

    def test_an_argv_naming_another_program_is_refused(self) -> None:
        for head in ("sh", "/bin/sh", "curl", "python3", ""):
            with self.subTest(head=head):
                with self.assertRaises(models.CatalogError) as caught:
                    models.read_catalog(self._doc([head, "-c", "id"]))
                self.assertIn("only 'kilix' is accepted", str(caught.exception))

    def test_unexpected_options_are_refused(self) -> None:
        with self.assertRaises(models.CatalogError) as caught:
            models.read_catalog(self._doc(["kilix", "stt", "--exec", "id"]))
        self.assertIn("only --install and --default", str(caught.exception))

    def test_a_non_argv_shape_is_refused(self) -> None:
        for argv in ("kilix stt", [], [["kilix"]], [1], {"a": 1}, None):
            with self.subTest(argv=argv):
                with self.assertRaises(models.CatalogError):
                    models.read_catalog(self._doc(argv))

    def test_control_characters_are_refused(self) -> None:
        with self.assertRaises(models.CatalogError) as caught:
            models.read_catalog(self._doc(["kilix", "stt\nid"]))
        self.assertIn("control character", str(caught.exception))


class R2Findings3And5TestCase(unittest.TestCase):
    """R2 findings 3 and 5, the parts that are code rather than judgement."""

    def _cuda(self, **measured):
        base = {"host": "pleon", "date": "2026-09-13",
                "peak_vram_mib": 100, "peak_ram_mib": 64}
        base.update(measured)
        return {"schema": resources.RESOURCE_SCHEMA,
                "device_class": resources.DEVICE_CUDA, "measured": base}

    def test_a_profile_missing_ram_is_refused(self) -> None:
        # R2: "No subject refusal covers missing RAM measurement." Now there is.
        bad = self._cuda()
        del bad["measured"]["peak_ram_mib"]
        with self.assertRaises(resources.ResourceError) as caught:
            resources.validate(bad)
        self.assertIn("must report peak_ram_mib", str(caught.exception))

    def test_a_cpu_profile_missing_ram_is_refused(self) -> None:
        with self.assertRaises(resources.ResourceError):
            resources.validate({
                "schema": resources.RESOURCE_SCHEMA,
                "device_class": resources.DEVICE_CPU,
                "measured": {"host": "h", "date": "2026-09-13",
                             "model_bytes": 10}})

    def test_an_accelerator_missing_vram_is_still_refused(self) -> None:
        bad = self._cuda()
        del bad["measured"]["peak_vram_mib"]
        with self.assertRaises(resources.ResourceError) as caught:
            resources.validate(bad)
        self.assertIn("must report peak_vram_mib", str(caught.exception))

    def test_a_complete_profile_is_accepted(self) -> None:          # control
        self.assertEqual(resources.validate(self._cuda())["device_class"], "cuda")

    def test_an_unmapped_measured_demand_refuses_rather_than_being_dropped(self) -> None:
        # R2: the comment promised refusal for an unmapped demand while the
        # dictionary filter silently dropped it. Simulate a future figure by
        # declaring one the map does not know.
        profile = self._cuda()
        with mock.patch.object(resources, "_MEASURED_INTS",
                               resources._MEASURED_INTS + ("peak_npu_mib",)):
            profile["measured"]["peak_npu_mib"] = 5
            self.assertFalse(resources.fits(profile, available_vram_mib=1000,
                                            available_ram_mib=1000))

    def test_the_audio_ceilings_are_inclusive_as_documented(self) -> None:
        # R2: the comment said "at or over ... is refused" while the code used
        # `>`. The code was right; the comment is now corrected. Pin both ends.
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "voicelib", "protocol.py")).read()
        self.assertIn("The ceiling is INCLUSIVE", src)
        self.assertNotIn("Anything at or over this is refused", src)
        self.assertEqual(protocol.check_audio_bytes(protocol.MAX_AUDIO_BYTES),
                         protocol.MAX_AUDIO_BYTES)


class PayloadBindingTestCase(unittest.TestCase):
    """R2 finding 2: consent must bind the installed BYTES, not just the name."""

    def setUp(self) -> None:
        self.home = os.path.realpath(tempfile.mkdtemp(prefix="f104-payload-"))
        self.env = mock.patch.dict(os.environ, {"HOME": self.home}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.addCleanup(shutil.rmtree, self.home, True)

    def _install(self, model_id="small-en-us", body=b"payload"):
        from voicelib import paths
        root = paths.model_dir(model_id)
        for relative in models.REQUIRED_FILES[models.ENGINE_VOSK]:
            target = os.path.join(root, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(body)
        return root

    def test_absent_payload_yields_an_empty_digest(self) -> None:
        self.assertEqual(consent.payload_digest("small-en-us", models.ENGINE_VOSK), "")

    def test_an_installed_payload_yields_a_digest(self) -> None:
        self._install()
        digest = consent.payload_digest("small-en-us", models.ENGINE_VOSK)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_changed_bytes_change_the_digest(self) -> None:      # S03, the point
        self._install(body=b"one")
        before = consent.payload_digest("small-en-us", models.ENGINE_VOSK)
        self._install(body=b"two")
        after = consent.payload_digest("small-en-us", models.ENGINE_VOSK)
        self.assertNotEqual(before, after)
        # and therefore the consent digest changes too
        self.assertNotEqual(
            consent.capture_digest("small-en-us", "vosk", before),
            consent.capture_digest("small-en-us", "vosk", after))

    def test_truncation_changes_the_digest(self) -> None:
        self._install(body=b"aaaa")
        before = consent.payload_digest("small-en-us", models.ENGINE_VOSK)
        self._install(body=b"")
        self.assertNotEqual(before,
                            consent.payload_digest("small-en-us", models.ENGINE_VOSK))

    def test_equal_length_content_change_is_still_detected(self) -> None:
        # The reason there is no stat cache. Two writes of equal length in
        # quick succession share size AND st_mtime_ns -- measured -- so a
        # (size, mtime) key returns the previous digest for changed bytes,
        # which is exactly the substitution this digest exists to catch.
        self._install(body=b"one")
        before = consent.payload_digest("small-en-us", models.ENGINE_VOSK)
        self._install(body=b"two")          # same length, same tick
        self.assertNotEqual(before,
                            consent.payload_digest("small-en-us", models.ENGINE_VOSK))

    def test_unchanged_bytes_give_a_stable_digest(self) -> None:   # control
        self._install()
        a = consent.payload_digest("small-en-us", models.ENGINE_VOSK)
        b = consent.payload_digest("small-en-us", models.ENGINE_VOSK)
        self.assertEqual(a, b)

    def test_production_callers_supply_the_payload_digest(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("kilix-voiced", "kilix-stt"):
            with self.subTest(tool=name):
                source = open(os.path.join(root, name)).read()
                self.assertIn("payload_digest(model_id, engine)", source)
