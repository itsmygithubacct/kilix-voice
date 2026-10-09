"""Managed callers share the canonical grant while Whistle keeps its default."""
import time
from pathlib import Path
import tempfile
import unittest
from voicelib import audio, device_leases, microphone
import kilix_device_lease as canonical

class ClosureTests(unittest.TestCase):
    def test_adapter_is_the_same_canonical_authority(self):
        self.assertIs(device_leases.acquire, canonical.acquire)
        self.assertIs(device_leases.Lease, canonical.Lease)
        self.assertIs(device_leases.LeaseError, canonical.LeaseError)
        self.assertFalse(Path(canonical.__file__).resolve().is_relative_to(Path(audio.__file__).resolve().parent.parent))

    def test_legacy_and_managed_capture_are_explicit_distinct_lifetimes(self):
        old = audio.MicCapture({'audio':{'capture_cmd':['not-executed']}})
        self.assertIs(type(old), audio.MicCapture)
        self.assertIsNone(old._process)
        control = microphone.CaptureControl(deadline=time.monotonic()+5, locked=lambda:False)
        managed = audio.MicCapture({'audio':{'capture_cmd':['not-executed']}}, control=control)
        self.assertEqual(type(managed).__name__, 'ManagedMicCapture')
        self.assertIs(managed._control, control)
        self.assertIsNone(managed._process)

    def test_canonical_quarantine_cannot_be_bypassed_through_adapter(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = str(Path(scratch)/'leases')
            grant = canonical.acquire(job_id='one', workload='stt-job',device='cpu',
                deadline=time.monotonic()+3,namespace=path)
            grant.release(cleanup_complete=False)
            with self.assertRaises(device_leases.LeaseError) as caught:
                device_leases.acquire(job_id='two',workload='tts-utterance',device='cpu',
                    deadline=time.monotonic()+3,namespace=path)
            self.assertEqual(caught.exception.code,'unavailable')
