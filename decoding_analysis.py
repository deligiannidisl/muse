#!/usr/bin/env python3
"""
Task-vs-rest decoding analysis for consumer-fNIRS cognitive monitoring.

Reproduces, from the raw recordings, every classification result reported in
the paper:

  Table 2  personalized (within-subject) and universal (cross-subject) decoding
  Section 4.3  participant-level tests against chance, and the personalization gap
  Table 3 / Section 4.4  robustness of the universal regime under five variants

--------------------------------------------------------------------------
INPUT
--------------------------------------------------------------------------
A directory of per-session CSV files written by the recorder, named

    <Task>_<participant>_<CONFIG>.csv        e.g.  Arithm_S01_INNERONLY.csv

where <Task> is "Arithm" or "Verbal" and <CONFIG> is "INNERONLY" or
"INNERandOUTER".  Each file must contain the columns

    Timestamp, L_HbO, L_HbR, R_HbO, R_HbR, Marker

with Marker = 1 during a task block and 0 otherwise.  The recorder writes the
four hemoglobin signals already inverted (modified Beer-Lambert) and filtered,
so this script starts from them.

--------------------------------------------------------------------------
METHOD
--------------------------------------------------------------------------
Epochs      Six task epochs, taken as the 30 s from each task-block onset, and
            six matched rest epochs, taken as the 30 s ending at each onset.
            Signals are resampled onto a uniform 7 Hz grid per session to
            remove the effect of the variable acquisition rate (27-64 Hz).
Features    Mean and linear slope of each of the four signals: 8 per epoch.
Classifier  Shrinkage linear discriminant analysis (Ledoit-Wolf).  Feature
            standardization is fitted on training data only and applied to
            training and test sets.
Regimes     Personalized  leave-one-epoch-out within each participant.
            Universal     leave-one-subject-out across participants.
Inference   The participant is the unit of inference: per-participant
            accuracies are tested against 0.5 with a two-sided one-sample
            t-test, and the personalization gap with a paired t-test and a
            Wilcoxon signed-rank test.  Epoch-pooled binomial values are
            printed for comparability with common practice but are not used
            to support any claim.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python decoding_analysis.py <data_directory>
    python decoding_analysis.py <data_directory> --exclude S09 --out results

Requires: numpy, pandas, scipy, scikit-learn.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SIGNALS = ["L_HbO", "L_HbR", "R_HbO", "R_HbR"]
COLUMNS = ["Timestamp"] + SIGNALS + ["Marker"]
EPOCH_S = 30.0          # task and rest epoch length, seconds
RESAMPLE_HZ = 7.0       # uniform grid, per session
N_DRAWS = 200           # random calibration draws averaged per participant
CALIB_K = (2, 4)        # labelled calibration epochs for the calibrated variants
SEED = 12345

TASKS = {"arithmetic": "Arithm_*_%s.csv", "verbal": "Verbal_*_%s.csv"}


# --------------------------------------------------------------------------
# loading and epoching
# --------------------------------------------------------------------------
def load_session(path):
    """Return (time, signals, marker) resampled onto a uniform grid."""
    df = pd.read_csv(path, usecols=COLUMNS)
    t = df.Timestamp.values.astype(float)
    Y = df[SIGNALS].values.astype(float)
    m = df.Marker.values.astype(int)
    grid = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
    Yr = np.column_stack([np.interp(grid, t, Y[:, k]) for k in range(Y.shape[1])])
    mr = (np.interp(grid, t, m.astype(float)) > 0.5).astype(int)
    return grid, Yr, mr


def block_onsets(t, m):
    """Onset time of each task block."""
    changes = np.where(np.diff(m) != 0)[0]
    return [t[i + 1] for i in changes if m[i + 1] == 1]


def epoch_features(t, Y, t0, t1):
    """Mean and linear slope of each signal over [t0, t1)."""
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
    tt, seg = t[i0:i1], Y[i0:i1]
    if len(tt) < 2:
        raise ValueError(f"empty epoch at t={t0:.1f}s - check the Marker column")
    x = tt - tt.mean()
    den = (x * x).sum()
    feats = []
    for k in range(seg.shape[1]):
        y = seg[:, k]
        feats.append(y.mean())
        feats.append((x * y).sum() / den if den > 0 else 0.0)
    return feats


def session_epochs(path):
    """Return (X, y) for one session: 6 task epochs (y=1) then 6 rest (y=0)."""
    t, Y, m = load_session(path)
    onsets = block_onsets(t, m)
    if len(onsets) != 6:
        print(f"  warning: {os.path.basename(path)} has {len(onsets)} task blocks, expected 6")
    X, y = [], []
    for s in onsets:                                   # task: 30 s from onset
        X.append(epoch_features(t, Y, s, s + EPOCH_S)); y.append(1)
    for s in onsets:                                   # rest: 30 s ending at onset
        X.append(epoch_features(t, Y, s - EPOCH_S, s)); y.append(0)
    return np.array(X), np.array(y)


def load_task(data_dir, pattern, config, exclude):
    """Map participant -> (X, y) for one task and channel configuration."""
    data = {}
    for f in sorted(glob.glob(os.path.join(data_dir, pattern % config))):
        name = os.path.basename(f)
        subject = re.match(r"(?:Arithm|Verbal)_(.+)_%s\.csv" % config, name).group(1)
        if subject in exclude:
            continue
        data[subject] = session_epochs(f)
    return data


# --------------------------------------------------------------------------
# classifiers and normalization
# --------------------------------------------------------------------------
def make_classifier(kind):
    if kind == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0)
    if kind == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    raise ValueError(kind)


def subject_zscore(X):
    """Label-free per-participant standardization."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def response_vector(X, y):
    """Task-minus-rest mean feature vector: a participant's response pattern."""
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def balanced_draw(y, k, rng):
    """Indices of k calibration epochs, half task and half rest."""
    task = rng.choice(np.where(y == 1)[0], k // 2, replace=False)
    rest = rng.choice(np.where(y == 0)[0], k - k // 2, replace=False)
    return np.concatenate([task, rest])


# --------------------------------------------------------------------------
# decoding regimes
# --------------------------------------------------------------------------
def personalized(data):
    """Leave-one-epoch-out within each participant."""
    acc = {}
    for s, (X, y) in data.items():
        correct = 0
        for i in range(len(y)):
            train = np.arange(len(y)) != i
            scaler = StandardScaler().fit(X[train])
            clf = make_classifier("lda").fit(scaler.transform(X[train]), y[train])
            correct += int(clf.predict(scaler.transform(X[[i]]))[0] == y[i])
        acc[s] = correct / len(y)
    return acc


def universal(data, kind="lda", norm="global"):
    """Leave-one-subject-out. norm: 'global' or 'subj' (per-participant z-scoring)."""
    subjects = sorted(data)
    acc = {}
    for s in subjects:
        Xtr = np.vstack([subject_zscore(data[u][0]) if norm == "subj" else data[u][0]
                         for u in subjects if u != s])
        ytr = np.concatenate([data[u][1] for u in subjects if u != s])
        Xte = subject_zscore(data[s][0]) if norm == "subj" else data[s][0]
        yte = data[s][1]
        if norm == "global":
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        acc[s] = float((make_classifier(kind).fit(Xtr, ytr).predict(Xte) == yte).mean())
    return acc


def universal_sign_aligned(data, k, rng):
    """Per-participant z-scoring plus sign alignment from k labelled epochs."""
    subjects = sorted(data)
    acc = {}
    for s in subjects:
        others = [u for u in subjects if u != s]
        Xtr = np.vstack([subject_zscore(data[u][0]) for u in others])
        ytr = np.concatenate([data[u][1] for u in others])
        population_sign = np.sign(np.mean(
            [response_vector(subject_zscore(data[u][0]), data[u][1]) for u in others], axis=0))
        population_sign[population_sign == 0] = 1
        clf = make_classifier("lda").fit(Xtr, ytr)

        Xs, ys = subject_zscore(data[s][0]), data[s][1]
        draws = []
        for _ in range(N_DRAWS):
            cal = balanced_draw(ys, k, rng)
            rest = np.setdiff1d(np.arange(len(ys)), cal)
            subject_sign = np.sign(response_vector(Xs[cal], ys[cal]))
            subject_sign[subject_sign == 0] = 1
            flip = population_sign * subject_sign   # -1 where the participant disagrees
            draws.append((clf.predict(Xs[rest] * flip) == ys[rest]).mean())
        acc[s] = float(np.mean(draws))
    return acc


def universal_two_stage(data, k, rng, n_clusters=2):
    """Cluster training participants by response pattern, then classify within cluster."""
    subjects = sorted(data)
    acc = {}
    for s in subjects:
        others = [u for u in subjects if u != s]
        R = np.array([response_vector(subject_zscore(data[u][0]), data[u][1]) for u in others])
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(R)

        models = {}
        for c in range(n_clusters):
            members = [u for u, lab in zip(others, km.labels_) if lab == c]
            if not members:
                continue
            Xc = np.vstack([subject_zscore(data[u][0]) for u in members])
            yc = np.concatenate([data[u][1] for u in members])
            if len(np.unique(yc)) == 2:
                models[c] = make_classifier("lda").fit(Xc, yc)
        fallback = make_classifier("lda").fit(
            np.vstack([subject_zscore(data[u][0]) for u in others]),
            np.concatenate([data[u][1] for u in others]))

        Xs, ys = subject_zscore(data[s][0]), data[s][1]
        draws = []
        for _ in range(N_DRAWS):
            cal = balanced_draw(ys, k, rng)
            rest = np.setdiff1d(np.arange(len(ys)), cal)
            c = int(km.predict(response_vector(Xs[cal], ys[cal])[None, :])[0])
            clf = models.get(c, fallback)
            draws.append((clf.predict(Xs[rest]) == ys[rest]).mean())
        acc[s] = float(np.mean(draws))
    return acc


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def summarize(acc, n_epochs):
    a = np.array([acc[s] for s in sorted(acc)])
    t, p = stats.ttest_1samp(a, 0.5)
    successes = int(round(a.sum() * n_epochs))
    binom = stats.binomtest(successes, len(a) * n_epochs, 0.5, alternative="greater").pvalue
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)),
                t=float(t), df=len(a) - 1, p=float(p), p_binomial_pooled=float(binom))


def paired_gap(acc_a, acc_b):
    subjects = sorted(set(acc_a) & set(acc_b))
    a = np.array([acc_a[s] for s in subjects])
    b = np.array([acc_b[s] for s in subjects])
    d = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(d, zero_method="wilcox")
        w_stat, p_w = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, p_w = float("nan"), float("nan")
    sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))
    return dict(n=len(d), gap=float(d.mean()), sd=float(sd),
                ci=(float(d.mean() - half), float(d.mean() + half)),
                t=float(t), df=len(d) - 1, p=float(p),
                W=w_stat, p_wilcoxon=p_w,
                dz=float(d.mean() / sd) if sd > 0 else float("inf"))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="directory containing the per-session CSV files")
    ap.add_argument("--config", default="INNERONLY",
                    choices=["INNERONLY", "INNERandOUTER"],
                    help="channel configuration to analyse (default: INNERONLY)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="participant identifiers to exclude")
    ap.add_argument("--out", default="results", help="output prefix")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"not a directory: {args.data_dir}")
    rng = np.random.default_rng(SEED)
    results = {}

    for task, pattern in TASKS.items():
        data = load_task(args.data_dir, pattern, args.config, set(args.exclude))
        if not data:
            print(f"\n{task}: no files matched - skipping")
            continue
        n_epochs = len(next(iter(data.values()))[1])
        print(f"\n{'=' * 72}\n{task.upper()}  ({len(data)} participants, "
              f"{n_epochs} epochs each, {args.config})\n{'=' * 72}")

        variants = [
            ("Personalized (within-subject)",       lambda d: personalized(d)),
            ("Universal baseline (global z, LDA)",  lambda d: universal(d, "lda", "global")),
            ("(a) per-subject z-scoring",           lambda d: universal(d, "lda", "subj")),
            ("(b) + sign alignment, k = 2",         lambda d: universal_sign_aligned(d, 2, rng)),
            ("(b) + sign alignment, k = 4",         lambda d: universal_sign_aligned(d, 4, rng)),
            ("(c) random forest",                   lambda d: universal(d, "rf", "global")),
            ("(c) RBF-SVM",                         lambda d: universal(d, "svm", "global")),
            ("(d) two-stage cluster, k = 4",        lambda d: universal_two_stage(d, 4, rng)),
        ]

        print(f"{'regime / variant':38s} {'mean ± SD':>15s} {'t':>7s} {'p':>9s}")
        print("-" * 72)
        task_results, accs = {}, {}
        for name, fn in variants:
            acc = fn(data)
            accs[name] = acc
            s = summarize(acc, n_epochs)
            task_results[name] = dict(**s, per_subject={k: round(v, 4) for k, v in acc.items()})
            print(f"{name:38s} {s['mean']*100:6.1f} ± {s['sd']*100:4.1f}% "
                  f"{s['t']:7.2f} {s['p']:9.4f}  {stars(s['p'])}")

        gap = paired_gap(accs["Personalized (within-subject)"],
                         accs["Universal baseline (global z, LDA)"])
        task_results["personalization_gap"] = gap
        print(f"\npersonalization gap: {gap['gap']*100:+.1f} pp "
              f"(95% CI [{gap['ci'][0]*100:+.1f}, {gap['ci'][1]*100:+.1f}])")
        print(f"  paired t({gap['df']}) = {gap['t']:.2f}, p = {gap['p']:.4f}; "
              f"Wilcoxon W = {gap['W']:.0f}, p = {gap['p_wilcoxon']:.4f}; dz = {gap['dz']:.2f}")
        print("\nper-participant accuracies (personalized / universal baseline):")
        for s in sorted(data):
            print(f"  {s:12s} {accs['Personalized (within-subject)'][s]*100:5.1f}% / "
                  f"{accs['Universal baseline (global z, LDA)'][s]*100:5.1f}%")
        results[task] = task_results

    out = f"{args.out}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
