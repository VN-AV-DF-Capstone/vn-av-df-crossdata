"""Metrics for cross-dataset scoring.

Ranking (threshold-free): AUC with a subject-level bootstrap CI, AP, EER, TPR at 1 % / 5 % FPR
(threshold set on the target reals: an oracle that shows the best achievable operating point).

Operating points that a user would actually get:
- transfer: the VN-AV-DF threshold (5 % FPR on VN-AV-DF validation reals) applied unchanged;
  FPR on the target reals shows how far the calibration drifts across languages and datasets;
- calibrated: the threshold re-fitted on the reals of a subject-disjoint calibration part of the
  target (sampling.calibration_split), FPR / TPR reported on the remaining evaluation part.

Paired comparison: AUC(reference) - AUC(other) on the same videos, bootstrap over subjects.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

LABEL_KEY = {"ai": "label", "visual": "synthetic_visual", "audio": "synthetic_audio"}


def target(row, task):
    return int(bool(row[LABEL_KEY[task]]))


def auc(labels, scores):
    labels = np.asarray(labels)
    return float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None


def eer(labels, scores):
    if len(set(labels)) != 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def rates(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    flagged = scores > threshold
    return {
        "threshold": float(threshold),
        "fpr": float(flagged[labels == 0].mean()) if (labels == 0).any() else None,
        "tpr": float(flagged[labels == 1].mean()) if (labels == 1).any() else None,
    }


def oracle(labels, scores, fpr_target):
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    reals = scores[labels == 0]
    if not len(reals) or not (labels == 1).any():
        return None
    return rates(labels, scores, np.quantile(reals, 1 - fpr_target))


def clusters_of(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["subject"]].append(i)
    return [np.asarray(v) for v in groups.values()]


def bootstrap(rows, labels, columns, replicates, seed):
    """Subject-level bootstrap of AUC for each score column → {name: [lo, hi]} and the draws."""
    labels = np.asarray(labels)
    groups = clusters_of(rows)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in columns}
    for _ in range(replicates):
        pick = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        y = labels[pick]
        if len(set(y.tolist())) < 2:
            continue
        for name, scores in columns.items():
            draws[name].append(roc_auc_score(y, np.asarray(scores)[pick]))
    ci = {n: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d else None
          for n, d in draws.items()}
    return ci, {n: np.asarray(d) for n, d in draws.items()}


def task_block(rows, scores, task, threshold, fpr_targets, replicates, seed):
    """All metrics of one model on one task; `scores` aligned with `rows`."""
    labels = [target(r, task) for r in rows]
    if len(set(labels)) < 2:
        return None
    scores = np.asarray(scores, float)
    ci, _ = bootstrap(rows, labels, {"auc": scores}, replicates, seed)
    out = {
        "n": len(rows),
        "n_fake": int(sum(labels)),
        "n_subjects": len({r["subject"] for r in rows}),
        "auc": auc(labels, scores),
        "auc_ci": ci["auc"],
        "ap": float(average_precision_score(labels, scores)),
        "eer": eer(labels, scores),
        "oracle": {f"fpr_{t:g}": oracle(labels, scores, t) for t in fpr_targets},
        "transfer": rates(labels, scores, threshold) if threshold is not None else None,
    }
    calib = [i for i, r in enumerate(rows) if r.get("part") == "calibration" and not labels[i]]
    evaluation = [i for i, r in enumerate(rows) if r.get("part") != "calibration"]
    if len(calib) >= 20 and evaluation:
        ev_labels = [labels[i] for i in evaluation]
        out["calibrated"] = {
            f"fpr_{t:g}": rates(ev_labels, scores[evaluation], np.quantile(scores[calib], 1 - t))
            for t in fpr_targets
        }
        out["calibration_reals"] = len(calib)
    return out


def group_auc(rows, scores, key, task="ai"):
    """AUC of each fake group (rows[key]) against all reals of the same task."""
    labels = np.asarray([target(r, task) for r in rows])
    scores = np.asarray(scores, float)
    reals = labels == 0
    out = {}
    for value in sorted({str(r.get(key)) for r, y in zip(rows, labels) if y}):
        fake = np.asarray([bool(y) and str(r.get(key)) == value for r, y in zip(rows, labels)])
        keep = reals | fake
        if fake.any() and reals.any():
            out[value] = {"auc": auc(labels[keep], scores[keep]), "n_fake": int(fake.sum())}
    return out


def subset_auc(rows, scores, sets):
    """MAVOS-DD test sets: AUC on the rows whose `subset` belongs to each set."""
    labels = np.asarray([r["label"] for r in rows])
    scores = np.asarray(scores, float)
    out = {}
    for name, parts in sets.items():
        keep = np.asarray([r.get("subset") in parts for r in rows])
        if keep.any() and len(set(labels[keep].tolist())) == 2:
            out[name] = {"auc": auc(labels[keep], scores[keep]), "n": int(keep.sum())}
    return out


def paired(rows, labels, reference, other, replicates, seed):
    """AUC(reference) - AUC(other) on the same rows, subject bootstrap → delta, CI, P(delta <= 0)."""
    _, draws = bootstrap(rows, labels, {"ref": reference, "other": other}, replicates, seed)
    diff = draws["ref"] - draws["other"]
    if not len(diff):
        return None
    return {
        "delta": auc(labels, reference) - auc(labels, other),
        "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "p_le_0": float((diff <= 0).mean()),
    }


def roc_points(labels, scores, points=200):
    fpr, tpr, _ = roc_curve(labels, scores)
    grid = np.linspace(0, 1, points)
    return grid.tolist(), np.interp(grid, fpr, tpr).tolist()
