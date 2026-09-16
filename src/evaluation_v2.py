"""IPW cumulative gain, exact tie-safe areas, paired stratified bootstrap.

G(q) is normalized by the ENTIRE evaluation population. Within a tied score
block, fractional selection means random tie-breaking in expectation.
"""
from itertools import combinations
import numpy as np


def _validate(scores, t, y, p=None):
    arrays = [np.asarray(v, dtype=float) for v in (scores, t, y)]
    if any(v.ndim != 1 for v in arrays):
        raise ValueError('scores, t and y must be 1D')
    s, t, y = arrays
    if not len(s) or len(s) != len(t) or len(s) != len(y):
        raise ValueError('arrays must have equal nonzero length')
    if not all(np.isfinite(v).all() for v in arrays):
        raise ValueError('inputs must be finite')
    if not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('t and y must be binary')
    if len(np.unique(t)) != 2:
        raise ValueError('both treatment arms are required')
    p = float(t.mean()) if p is None else float(p)
    if not np.isfinite(p) or not 0 < p < 1:
        raise ValueError('scalar p must be strictly between 0 and 1')
    return s, t, y, p


def _prepare(scores, z):
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    return order, starts, z[order]


def _calculate(prepared, weights=None, with_curve=False):
    order, starts, z = prepared
    w = np.ones(len(order)) if weights is None else weights[order].astype(float)
    counts = np.add.reduceat(w, starts)
    contributions = np.add.reduceat(w * z, starts)
    keep = counts > 0  # an entire tie block can disappear in a bootstrap replicate
    counts, contributions = counts[keep], contributions[keep]
    n = counts.sum()
    fractions = np.r_[0., np.cumsum(counts) / n]
    gain = np.r_[0., np.cumsum(contributions) / n]
    ate = float(gain[-1])
    # Integrate at ALL score-block endpoints, not on the coarse plotting grid.
    auuc = float(np.sum(np.diff(fractions) * (gain[1:] + gain[:-1]) / 2))
    metrics = {'auuc': auuc, 'qini_area': auuc - ate / 2, 'overall_ate': ate}
    random = {'auuc': ate / 2, 'qini_area': 0., 'overall_ate': ate}
    for pct in (10, 20, 30):
        q = pct / 100
        g = float(np.interp(q, fractions, gain))
        metrics[f'top{pct}_policy_gain'] = g
        metrics[f'top{pct}_selected_ate'] = g / q
        metrics[f'top{pct}_gain_over_random'] = g - q * ate
        random[f'top{pct}_policy_gain'] = q * ate
        random[f'top{pct}_selected_ate'] = ate
        random[f'top{pct}_gain_over_random'] = 0.
    result = {'metrics': metrics, 'random_expectation': random}
    if with_curve:
        grid = np.linspace(0, 1, 101)
        g = np.interp(grid, fractions, gain)
        result['curve'] = {'fraction': grid.tolist(), 'gain': g.tolist(),
                           'qini': (g - grid * ate).tolist()}
    return result


def evaluate(scores, t, y, p=None):
    s, t, y, p = _validate(scores, t, y, p)
    z = t*y/p - (1-t)*y/(1-p)
    result = _calculate(_prepare(s, z), with_curve=True)
    result['evaluation_propensity'] = p
    return result


def paired_bootstrap(scores_by_model, t, y, n_bootstrap=200, seed=2027, p=None):
    """Percentile CIs conditional on fitted scores, fixed arm counts; no refitting.

    Every model shares the same resampled row multiplicities. Sorted score
    groups are reused; areas/top-K cutoffs are recomputed with multiplicities.
    The theoretical random baseline is included, not a noisy single shuffle.
    """
    if not scores_by_model or int(n_bootstrap) != n_bootstrap or n_bootstrap < 2:
        raise ValueError('models and at least 2 bootstrap replicates required')
    prepared = {}
    for name, scores in scores_by_model.items():
        if name == 'random_expectation':
            raise ValueError('random_expectation is reserved')
        s, t, y, propensity = _validate(scores, t, y, p)
        z = t*y/propensity - (1-t)*y/(1-propensity)
        prepared[name] = _prepare(s, z)
    metrics = ('qini_area', 'top20_policy_gain', 'top20_gain_over_random')
    values = {name: {m: [] for m in metrics}
              for name in list(prepared) + ['random_expectation']}
    arms = [np.flatnonzero(t == a) for a in (0, 1)]
    rng = np.random.default_rng(seed)
    for _ in range(int(n_bootstrap)):
        sampled = np.concatenate([rng.choice(a, size=len(a), replace=True) for a in arms])
        weights = np.bincount(sampled, minlength=len(t))
        for name, prep in prepared.items():
            result = _calculate(prep, weights)
            for metric in metrics:
                values[name][metric].append(result['metrics'][metric])
        for metric in metrics:
            values['random_expectation'][metric].append(result['random_expectation'][metric])
    def interval(v):
        return np.quantile(v, [.025, .975]).tolist()
    pairs = {}
    for a, b in combinations(values, 2):
        pairs[f'{a}-{b}'] = {m: interval(np.array(values[a][m]) - values[b][m]) for m in metrics}
    return {'n_bootstrap': int(n_bootstrap), 'seed': int(seed),
            'evaluation_propensity': propensity,
            'ci': {name: {m: interval(v) for m, v in val.items()} for name, val in values.items()},
            'pairwise_difference_ci': pairs,
            'note': '95% stratified paired percentile CI, conditional on fixed fitted models and arm counts. No refitting, model selection correction, or final holdout claim.'}
