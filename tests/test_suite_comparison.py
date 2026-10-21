import unittest
from tests.suite_comparison import compare_runs


class Comparison(unittest.TestCase):
    def run_result(self, *identities):
        return dict(exit=1 if identities else 0, identities=identities,
                    summary=['Ran 100 tests in 1s'])

    def test_different_clipboard_method_is_unresolved(self):
        baseline = 'FAIL: test_pending_copy (test_clipboard_transport.ClipboardTransportTests.test_pending_copy)'
        candidate = 'FAIL: test_oversized_incr (test_clipboard_transport.ClipboardTransportTests.test_oversized_incr)'
        result = compare_runs([self.run_result(baseline)] * 3,
                              [self.run_result(candidate), self.run_result(), self.run_result()])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['unresolved'], [candidate])

    def test_one_baseline_occurrence_permits_that_exact_difference(self):
        identity = 'FAIL: case (module.Class.case) (flag="-h")'
        result = compare_runs([self.run_result(), self.run_result(identity), self.run_result()],
                              [self.run_result(identity)] * 3)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['unresolved'], [])

    def test_missing_or_incomplete_runs_cannot_pass(self):
        with self.assertRaises(ValueError):
            compare_runs([self.run_result()] * 2, [self.run_result()] * 3)
        with self.assertRaises(ValueError):
            compare_runs([self.run_result()] * 3, [dict(exit='timeout')] * 3)
