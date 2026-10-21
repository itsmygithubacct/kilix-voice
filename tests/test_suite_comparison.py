import unittest
from tests.suite_comparison import compare_runs


class Comparison(unittest.TestCase):
    def run_result(self, *identities):
        failures = sum(identity.startswith('FAIL:') for identity in identities)
        errors = sum(identity.startswith('ERROR:') for identity in identities)
        counts = ', '.join(f'{key}={value}' for key, value in
                           [('failures', failures), ('errors', errors)] if value)
        return dict(exit=1 if identities else 0, identities=identities,
                    summary=['Ran 100 tests in 1s', f'FAILED ({counts})' if identities else 'OK'])

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

    def test_rejects_missing_and_inconsistent_final_summaries(self):
        failure = 'FAIL: case (module.Class.case)'
        error = 'ERROR: case (module.Class.case)'
        bad = [
            dict(exit=1, identities=[], summary=['Ran 10 tests in 1s', 'FAILED (failures=1)']),
            dict(exit=0, identities=[], summary=['Ran 10 tests in 1s']),
            dict(exit=1, identities=[failure], summary=['Ran 10 tests in 1s', 'OK']),
            dict(exit=0, identities=[failure], summary=['Ran 10 tests in 1s', 'FAILED (failures=1)']),
            dict(exit=1, identities=[error], summary=['Ran 10 tests in 1s', 'FAILED (failures=1)']),
            dict(exit=1, identities=[failure], summary=['Ran 10 tests in 1s', 'FAILED (failures=2)']),
            dict(exit=0, identities=[], summary=['Ran 10 tests in 1s', 'OK', 'FAILED (errors=1)']),
            dict(exit=0, identities=[], summary=['OK', 'Ran 10 tests in 1s']),
            dict(exit=0, identities=[], summary=['Ran nonsense', 'OK']),
            dict(exit=0, identities=[], summary=['Ran 10 tests in 1s', 'OK (skipped=-1)']),
            dict(exit=1, identities=[failure], summary=['Ran 10 tests in 1s', 'FAILED (failures=1, failures=1)']),
            dict(exit=1, identities=[failure], summary=['Ran 10 tests in 1s', 'FAILED (failures=1, unknown=1)']),
            dict(exit=1, identities=['unparsed failure'], summary=['Ran 10 tests in 1s', 'FAILED (failures=1)']),
        ]
        for record in bad:
            for baseline in (False, True):
                with self.subTest(record=record, baseline=baseline):
                    good = [self.run_result()] * 3
                    with self.assertRaises(ValueError):
                        compare_runs([record] * 3 if baseline else good,
                                     good if baseline else [record] * 3)

    def test_complete_skip_and_subtest_summaries_are_accepted(self):
        identity = 'FAIL: case (module.Class.case) (value=0)'
        record = dict(exit=1, identities=[identity, identity],
                      summary=['Ran 1 test in 0.010s', 'FAILED (failures=2)'])
        self.assertTrue(compare_runs([record] * 3, [record] * 3)['accepted'])

    def test_skipped_subtests_can_outnumber_test_methods(self):
        record = dict(exit=0, identities=[], summary=['Ran 1 test in 0.001s', 'OK (skipped=3)'])
        self.assertTrue(compare_runs([record] * 2, [record] * 2)['accepted'])
        record = dict(exit=0, identities=[],
                      summary=['Ran 10 tests in 1.000s', 'OK (skipped=2, expected failures=1)'])
        self.assertTrue(compare_runs([record] * 3, [record] * 3)['accepted'])
