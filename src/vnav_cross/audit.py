"""VN-AV-DF leakage audit: can the in-domain test numbers be explained by a leak?

Checks (all from the released manifests and logs, no model needed):
1. identity / source overlap between splits, over all parts together: speaker_id, YouTube source
   video (source_id), real clip (source_clip_id), file hash (sha256), and the audio donor
   (audio_source_id: sham / donor / voice clips borrow speech from another real video);
2. sentence overlap: the same transcript in train and test (AI-voice text);
3. test reuse: how many times each checkpoint touched the test split (test-log.json, written by
   training/03_evaluate.py every time a model is tested) and how many distinct sessions did;
4. naive shortcuts: AUC of single hand-crafted features (leading / trailing silence, loudness,
   duration, ...) for AI vs not-AI, from shortcut_audit.json of every part (generation/audit.py),
   plus duration from the manifest for every split. AUC far from 0.5 means a model could separate
   classes without looking at lips or voices.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

SPLITS = ("train", "validation", "test")
KEYS = ("speaker_id", "source_id", "source_clip_id", "sha256", "audio_source_id")


def read_manifests(paths):
    rows = []
    for path in paths:
        part = Path(path).parent.name
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_part"] = part
                rows.append(r)
    return rows


def split_counts(rows):
    out = defaultdict(Counter)
    for r in rows:
        out[r["_part"]][r["split"]] += 1
    speakers = defaultdict(set)
    for r in rows:
        speakers[r["split"]].add(r["speaker_id"])
    return {"files": {p: dict(c) for p, c in sorted(out.items())},
            "speakers": {s: len(v) for s, v in speakers.items()}}


def overlaps(rows, keys=KEYS, examples=5):
    """For each key: values seen in more than one split (count per split pair, a few examples)."""
    out = {}
    for key in keys:
        where = defaultdict(set)
        for r in rows:
            value = r.get(key)
            if value:
                where[value].add(r["split"])
        pairs = Counter()
        shown = defaultdict(list)
        for value, splits in where.items():
            if len(splits) > 1:
                pair = "+".join(s for s in SPLITS if s in splits)
                pairs[pair] += 1
                if len(shown[pair]) < examples:
                    shown[pair].append(value)
        out[key] = {"values": len(where), "shared": sum(pairs.values()),
                    "by_pair": dict(pairs), "examples": dict(shown)}
    return out


def donor_crossing(rows):
    """Clips whose borrowed audio (audio_source_id) is a real video of another split."""
    split_of = defaultdict(set)
    for r in rows:
        if r.get("cell") == "real":
            split_of[r["source_id"]].add(r["split"])
    crossing = Counter()
    for r in rows:
        donor = r.get("audio_source_id")
        if donor and donor != r["source_id"] and split_of.get(donor) and r["split"] not in split_of[donor]:
            crossing[f"{r['cell']}:{r['split']}<-{'+'.join(sorted(split_of[donor]))}"] += 1
    borrowed = sum(1 for r in rows if r.get("audio_source_id") and r["audio_source_id"] != r["source_id"])
    return {"borrowed_audio_clips": borrowed, "crossing_splits": dict(crossing)}


def transcript_overlap(rows):
    by = defaultdict(set)
    for r in rows:
        text = (r.get("transcript") or "").strip().lower()
        if text:
            by[text].add(r["split"])
    shared = Counter("+".join(s for s in SPLITS if s in v) for v in by.values() if len(v) > 1)
    return {"sentences": len(by), "shared": sum(shared.values()), "by_pair": dict(shared)}


TRAINING_FREE = ("speechforensics",)  # no weights: the same "checkpoint" hash in every round


def test_reuse(log_paths):
    """test-log.json of every round: uses per checkpoint and per model name. Training-free
    detectors are left out of the per-checkpoint count (their hash never changes)."""
    entries = []
    for path in log_paths:
        path = Path(path)
        if path.is_file():
            for e in json.loads(path.read_text(encoding="utf-8")):
                entries.append({**e, "round": path.parent.name})
    per_checkpoint = Counter(
        e["checkpoint"] for e in entries if e["model"].rsplit("_seed", 1)[0] not in TRAINING_FREE
    )
    per_model = defaultdict(set)
    for e in entries:
        per_model[e["model"].rsplit("_seed", 1)[0]].add((e["round"], e["checkpoint"]))
    return {
        "entries": len(entries),
        "rounds": sorted({e["round"] for e in entries}),
        "sessions": len({e.get("session") for e in entries}),
        "checkpoints": len(per_checkpoint),
        "checkpoints_tested_more_than_once": sum(1 for c in per_checkpoint.values() if c > 1),
        "max_test_use": max([int(e.get("test_use", 1)) for e in entries] or [0]),
        "per_model": {m: len(v) for m, v in sorted(per_model.items())},
        "table": [[e["round"], e["model"], e.get("session"), e.get("time"), e.get("test_use")] for e in entries],
    }


def shortcut_summary(audit_paths, flag=(0.3, 0.7)):
    out = {}
    for path in audit_paths:
        path = Path(path)
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        flagged = {f: v for f, v in (data.get("overall") or {}).items() if not flag[0] <= v <= flag[1]}
        worst = {}
        for cell, feats in (data.get("by_cell_vs_real") or {}).items():
            for f, v in feats.items():
                if abs(v - 0.5) > abs(worst.get(f, (None, 0.5))[1] - 0.5):
                    worst[f] = (cell, v)
        out[path.parent.name] = {"samples": data.get("samples"), "overall": data.get("overall"),
                                 "flagged_overall": flagged,
                                 "most_extreme_cell": {f: {"cell": c, "auc": v} for f, (c, v) in worst.items()}}
    return out


def duration_auc(rows):
    out = {}
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        labels = [r["label"] for r in part]
        if len(set(labels)) == 2:
            out[split] = float(roc_auc_score(labels, [r["duration_s"] for r in part]))
    return out


def run_audit(manifests, test_logs=(), shortcut_audits=()):
    rows = read_manifests(manifests)
    over = overlaps(rows)
    result = {
        "manifests": [str(p) for p in manifests],
        "rows": len(rows),
        "splits": split_counts(rows),
        "overlap": over,
        "donor_audio": donor_crossing(rows),
        "transcripts": transcript_overlap(rows),
        "test_reuse": test_reuse(test_logs),
        "shortcuts": shortcut_summary(shortcut_audits),
        "duration_auc": duration_auc(rows),
    }
    leaks = [k for k in ("speaker_id", "source_id", "source_clip_id", "sha256") if over[k]["shared"]]
    result["verdict"] = {
        "identity_or_source_leak": leaks,
        "donor_audio_crossing": sum(result["donor_audio"]["crossing_splits"].values()),
        "repeated_test_use": result["test_reuse"]["checkpoints_tested_more_than_once"],
    }
    return result
