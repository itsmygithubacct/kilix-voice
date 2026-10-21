"""Exact failure-identity comparison for three baseline/candidate suite runs."""


def compare_runs(baselines, candidates):
    if len(baselines) != 3 or len(candidates) != 3:
        raise ValueError('three complete runs of each revision are required')
    for run in [*baselines, *candidates]:
        if run.get('exit') not in (0, 1) or not any(
                line.startswith('Ran ') for line in run.get('summary', [])):
            raise ValueError('incomplete suite run')
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
