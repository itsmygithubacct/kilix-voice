"""Opt-in wall-clock check, run on one CPU without the loaded repetition job."""
import os
import statistics
import subprocess
import time
import unittest
from voicelib.processes import GroupPopen


@unittest.skipUnless(os.environ.get('VOICE_TEST_SPAWN_BUDGET') == '1',
                     'requires an explicitly selected idle CPU')
class SpawnBudget(unittest.TestCase):
    def test_added_median_at_most_fifteen_ms(self):
        samples = {subprocess.Popen: [], GroupPopen: []}
        for i in range(23):
            for factory in samples:
                start = time.perf_counter()
                child = factory(['/bin/true'], start_new_session=True)
                elapsed = time.perf_counter() - start
                child.wait(timeout=3)
                if i >= 3:
                    samples[factory].append(elapsed)
        added = statistics.median(samples[GroupPopen]) - statistics.median(samples[subprocess.Popen])
        self.assertLessEqual(added, .015, f'launcher added {added * 1000:.2f} ms median')
