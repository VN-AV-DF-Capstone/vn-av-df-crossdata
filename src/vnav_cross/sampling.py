"""Stratified, seeded subsampling and the subject-disjoint calibration split.

Each dataset is far larger than one Kaggle GPU week can extract (AV-HuBERT Large on every video),
so we score a fixed, stratified sample. The sample is a pure function of (index, spec, seed):
re-running gives the same videos, and `sample_stats` records both the full and the sampled
counts per stratum so the report can state what was left out.
"""

import hashlib
import random
from collections import Counter, defaultdict


def stratum(row, keys):
    return tuple(str(row.get(k)) for k in keys)


def sample(rows, spec, seed):
    """Up to `per_stratum` rows per stratum (keys `strata`); `max_per_subject` > 0 caps how many
    rows of one subject a stratum takes, so a few identities cannot dominate a class."""
    keys = list(spec.get("strata") or ["category"])
    per, cap = int(spec.get("per_stratum") or 0), int(spec.get("max_per_subject") or 0)
    groups = defaultdict(list)
    for r in rows:
        groups[stratum(r, keys)].append(r)
    picked = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda r: r["sample_id"])
        random.Random(f"{seed}:{'|'.join(key)}").shuffle(items)
        taken, per_subject = [], Counter()
        for r in items:
            if per and len(taken) >= per:
                break
            if cap and per_subject[r["subject"]] >= cap:
                continue
            per_subject[r["subject"]] += 1
            taken.append(r)
        picked += taken
    return sorted(picked, key=lambda r: r["sample_id"])


def sample_stats(full, picked, keys):
    total = Counter(stratum(r, keys) for r in full)
    kept = Counter(stratum(r, keys) for r in picked)
    return {
        "strata": list(keys),
        "full": len(full),
        "sampled": len(picked),
        "rows": [
            {"stratum": "|".join(k), "full": total[k], "sampled": kept.get(k, 0)}
            for k in sorted(total)
        ],
        "subjects_full": len({r["subject"] for r in full}),
        "subjects_sampled": len({r["subject"] for r in picked}),
    }


def calibration_split(rows, fraction, seed):
    """Mark ~`fraction` of the subjects as "calibration" (re-fit thresholds on the target domain),
    the rest as "evaluation". Split by subject so no identity is on both sides."""
    for r in rows:
        digest = hashlib.sha1(f"{seed}:{r['subject']}".encode()).hexdigest()
        r["part"] = "calibration" if int(digest[:8], 16) / 0xFFFFFFFF < fraction else "evaluation"
    return rows
