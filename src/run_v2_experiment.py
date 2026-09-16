"""Reproducible HGB S/T/X comparison on a fixed development validation set."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

from evaluation_v2 import evaluate, paired_bootstrap

FEATURES = [f'f{i}' for i in range(12)]
OUTCOME_PARAMS = dict(max_iter=50, max_leaf_nodes=15, learning_rate=.1,
                      l2_regularization=1., early_stopping=False)
EFFECT_PARAMS = dict(max_iter=80, max_leaf_nodes=15, learning_rate=.08,
                     l2_regularization=1., early_stopping=False)


def outcome(seed):
    return HistGradientBoostingClassifier(**OUTCOME_PARAMS, random_state=seed)


def effect(seed):
    return HistGradientBoostingRegressor(**EFFECT_PARAMS, random_state=seed)


def sample_training(pool, t, train_size, fraction, seed):
    """Select by treatment ONLY; total n stays fixed across fraction scenarios."""
    if train_size is None:
        if fraction is not None:
            raise ValueError('--treated-fraction requires --train-size')
        return pool.copy()
    if train_size < 12 or train_size > len(pool):
        raise ValueError('train-size must be >=12 and fit inside training pool')
    if fraction is None or not 0 < fraction < 1:
        raise ValueError('train-size requires 0 < treated-fraction < 1')
    n1 = round(train_size * fraction)
    n0 = train_size - n1
    rng = np.random.default_rng(seed)
    arms = [pool[t[pool] == arm] for arm in (0, 1)]
    if n0 < 3 or n1 < 3 or n0 > len(arms[0]) or n1 > len(arms[1]):
        raise ValueError('requested treatment composition is infeasible')
    selected = np.r_[rng.choice(arms[0], n0, replace=False),
                     rng.choice(arms[1], n1, replace=False)]
    return rng.permutation(selected)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def group_counts(t, y):
    return {str(a): {'n': int((t == a).sum()),
                     'conversions': int(y[t == a].sum()),
                     'conversion_rate': float(y[t == a].mean())} for a in (0, 1)}


def validate_training(t, y, folds=3):
    for arm in (0, 1):
        labels = y[t == arm]
        counts = np.bincount(labels.astype(int), minlength=2)
        if len(labels) < folds or counts.min() < folds:
            raise ValueError(f'arm {arm} needs >= {folds} positives and negatives for cross-fitting')


def fit_scores(Xp, tp, yp, Xv, seed):
    validate_training(tp, yp)
    tmodels = {}
    for arm in (0, 1):
        mask = tp == arm
        tmodels[arm] = outcome(seed + arm).fit(Xp[mask], yp[mask])
    mu0 = tmodels[0].predict_proba(Xv)[:, 1]
    mu1 = tmodels[1].predict_proba(Xv)[:, 1]
    sm = outcome(seed + 10).fit(np.c_[Xp, tp], yp)
    s1 = sm.predict_proba(np.c_[Xv, np.ones(len(Xv))])[:, 1]
    s0 = sm.predict_proba(np.c_[Xv, np.zeros(len(Xv))])[:, 1]
    mu0_oof = np.full(len(tp), np.nan)
    mu1_oof = np.full(len(tp), np.nan)
    coverage = np.zeros(len(tp), dtype=int)
    fold_audit = []
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for fitpos, valpos in skf.split(Xp, tp):
        # train/held-out positions are local to Xp; no validation labels enter fit.
        if np.intersect1d(fitpos, valpos).size:
            raise RuntimeError('cross-fitting fold overlap')
        for arm, destination in ((0, mu0_oof), (1, mu1_oof)):
            sub = fitpos[tp[fitpos] == arm]
            if len(np.unique(yp[sub])) != 2:
                raise ValueError('fold lacks a binary outcome class; increase training sample')
            m = outcome(seed + 20 + arm).fit(Xp[sub], yp[sub])
            destination[valpos] = m.predict_proba(Xp[valpos])[:, 1]
        coverage[valpos] += 1
        fold_audit.append({'fit_n': len(fitpos), 'heldout_n': len(valpos),
                           'fit_groups': group_counts(tp[fitpos], yp[fitpos])})
    if not ((coverage == 1).all() and np.isfinite(mu0_oof).all() and np.isfinite(mu1_oof).all()):
        raise RuntimeError('incomplete OOF coverage')
    d1 = yp[tp == 1] - mu0_oof[tp == 1]
    d0 = mu1_oof[tp == 0] - yp[tp == 0]
    m1 = effect(seed + 30).fit(Xp[tp == 1], d1)
    m0 = effect(seed + 31).fit(Xp[tp == 0], d0)
    p_train = float(tp.mean())
    scores = {
        'random_baseline': np.random.default_rng(seed).random(len(Xv)),
        't_learner_hgb': mu1 - mu0,
        's_learner_hgb': s1 - s0,
        'x_learner_hgb_crossfit': (1-p_train)*m1.predict(Xv)+p_train*m0.predict(Xv),
    }
    probability_predictions = {'t_learner_hgb': (mu0, mu1), 's_learner_hgb': (s0, s1)}
    return scores, probability_predictions, fold_audit


def fit_model(path, seed=2027, max_bootstrap=200, out='results/v2_main_seed2027.json',
              train_size=None, treated_fraction=None, split_seed=2027, threads=2):
    if max_bootstrap != 0 and max_bootstrap < 2:
        raise ValueError('bootstrap must be 0 (skip) or >=2')
    if threads < 1:
        raise ValueError('threads must be positive')
    df = pd.read_csv(path, usecols=FEATURES+['treatment', 'conversion'])
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError('non-finite data')
    if any(not df[c].isin([0, 1]).all() for c in ['treatment', 'conversion']):
        raise ValueError('treatment/conversion must be 0/1')
    X = df[FEATURES].to_numpy(dtype='float32')
    t = df.treatment.to_numpy(dtype='int8')
    y = df.conversion.to_numpy(dtype='int8')
    pool, val = train_test_split(np.arange(len(df)), test_size=.25, random_state=split_seed, stratify=t)
    selected = sample_training(pool, t, train_size, treated_fraction, seed)
    if np.intersect1d(selected, val).size:
        raise RuntimeError('train and validation overlap')
    Xp, tp, yp = X[selected], t[selected], y[selected]
    tv, yv = t[val], y[val]
    with threadpool_limits(limits=threads):
        scores, probabilities, folds = fit_scores(Xp, tp, yp, X[val], seed)
    evaluations = {name: evaluate(sc, tv, yv) for name, sc in scores.items()}
    boot = paired_bootstrap(scores, tv, yv, n_bootstrap=max_bootstrap, seed=seed) if max_bootstrap else None
    # Factual outcome calibration diagnostics only; X outputs effects, not a probability.
    diagnostics = {}
    for name, (p0, p1) in probabilities.items():
        pred = np.where(tv == 1, p1, p0)
        diagnostics[name] = {'brier_score': float(brier_score_loss(yv, pred)),
                             'log_loss': float(log_loss(yv, pred, labels=[0, 1])),
                             'predicted_mean_by_arm': {str(a): float(pred[tv == a].mean()) for a in (0, 1)}}
    result = {
        'version': 'v2.1', 'dataset': Path(path).name, 'dataset_sha256': file_sha256(path),
        'split_seed': split_seed, 'model_seed': seed, 'n': len(df),
        'train_pool_n': len(pool), 'train_n': len(selected), 'validation_n': len(val),
        'validation_indices_sha256': hashlib.sha256(val.astype('<i8').tobytes()).hexdigest(),
        'train_indices_sha256': hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
        'features': FEATURES, 'requested_treated_fraction': treated_fraction,
        'train_propensity': float(tp.mean()), 'validation_propensity': float(tv.mean()),
        'propensity_note': 'evaluation: validation arm-frequency plug-in; X mixing: sampled-training arm frequency, not target population propensity',
        'train_groups': group_counts(tp, yp), 'validation_groups': group_counts(tv, yv),
        'outcome_model': 'unweighted HistGradientBoostingClassifier', 'outcome_params': OUTCOME_PARAMS,
        'effect_model': 'HistGradientBoostingRegressor (X only)', 'effect_params': EFFECT_PARAMS,
        'cross_fitting_folds': 3, 'fold_audit': folds,
        'metrics': {k: v['metrics'] for k, v in evaluations.items()},
        'curves': {k: v['curve'] for k, v in evaluations.items()},
        'random_expectation': evaluations['random_baseline']['random_expectation'],
        'outcome_diagnostics': diagnostics, 'bootstrap': boot,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'threads': threads},
        'status': 'development validation; reused for model exploration; not untouched final test',
    }
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.json.part')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(destination)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path')
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--out', default='results/v2_main_seed2027.json')
    ap.add_argument('--bootstrap', type=int, default=200)
    ap.add_argument('--train-size', type=int)
    ap.add_argument('--treated-fraction', type=float)
    ap.add_argument('--split-seed', type=int, default=2027)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    start = time.monotonic()
    r = fit_model(a.path, a.seed, a.bootstrap, a.out, a.train_size, a.treated_fraction, a.split_seed, a.threads)
    print(json.dumps({'seed': a.seed, 'n_train': r['train_n'], 'treated_fraction': r['train_propensity'],
                      'qini': {k: v['qini_area'] for k, v in r['metrics'].items()},
                      'bootstrap': a.bootstrap, 'elapsed_sec': round(time.monotonic()-start, 1)}))
