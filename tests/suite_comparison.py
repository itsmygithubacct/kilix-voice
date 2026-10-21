"""Exact failure-identity comparison for repeated baseline/candidate suites.

Kept here so voice changes ship the validator used by downstream pin-comparison runners.
"""
import re


def _validate_run(run):
    summary = run.get('summary', [])
    identities = run.get('identities')
    if (type(run.get('exit')) is not int or run['exit'] not in (0, 1)
            or not isinstance(summary, (list, tuple)) or len(summary) != 2
            or not all(isinstance(line, str) for line in summary)
            or not isinstance(identities, (list, tuple))):
        raise ValueError('incomplete suite run')
    ran = re.fullmatch(r'Ran (\d+) tests? in \d+(?:\.\d+)?s', summary[0])
    final = re.fullmatch(r'(OK|FAILED)(?: \((.+)\))?', summary[1])
    if not ran or not final or int(ran[1]) == 0:
        raise ValueError('missing or malformed terminal summary')
    counts = {}
    if final[2]:
        for item in final[2].split(', '):
            count = re.fullmatch(r'(failures|errors|skipped|expected failures|unexpected successes)=(\d+)', item)
            if not count or count[1] in counts:
                raise ValueError('invalid or duplicate summary count')
            counts[count[1]] = int(count[2])
    parsed = {'FAIL': 0, 'ERROR': 0}
    for identity in identities:
        match = re.fullmatch(r'(FAIL|ERROR): \S[^\n]*', identity) if isinstance(identity, str) else None
        if not match:
            raise ValueError('unparsed failure identity')
        parsed[match[1]] += 1
    failed = final[1] == 'FAILED'
    if (run['exit'] != int(failed) or failed != bool(identities)
            or parsed['FAIL'] != counts.get('failures', 0)
            or parsed['ERROR'] != counts.get('errors', 0)
            or counts.get('unexpected successes', 0)
            or counts.get('expected failures', 0) > int(ran[1])):
        raise ValueError('exit status, identities and final summary disagree')
    # Subtest failures, errors and skips can outnumber test methods.


def compare_runs(baselines, candidates):
    if len(baselines) < 2 or len(baselines) != len(candidates):
        raise ValueError('at least two complete paired runs are required')
    for run in [*baselines, *candidates]:
        _validate_run(run)
    baseline_sets = [set(run['identities']) for run in baselines]
    candidate_sets = [set(run['identities']) for run in candidates]
    observed_baseline = set.union(*baseline_sets)
    observed_candidate = set.union(*candidate_sets)
    unresolved = observed_candidate - observed_baseline
    return dict(
        baseline_union=sorted(observed_baseline),
        candidate_union=sorted(observed_candidate),
        unresolved=sorted(unresolved),
        baseline_only=sorted(observed_baseline - observed_candidate),
        accepted=not unresolved,
    )
