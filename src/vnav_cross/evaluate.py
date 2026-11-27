"""Evaluation: every detector on the SAME videos of each dataset, every protocol variant.

- Common set: the videos that every primary detector scored (base condition). Coverage per class
  (sampled, media error, features missing, scored) is reported so dropped videos stay visible.
- Resolution gate (M3-R and BL-Ens-AV, i.e. models with both a visual and an audio score): a
  video whose source short side is below `gate_min_side` is judged by the audio score only, with
  the audio threshold, exactly as vn_av_df.training.evaluate.resolution_gate does in the demo.
  Both "gate on" (the shipped inference) and "gate off" are reported.
- Thresholds are compared through margins (score - threshold of the head actually used), so the
  transfer FPR stays correct when gated and ungated videos are mixed.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from vnav_cross import datasets as ds
from vnav_cross import metrics as M
from vnav_cross.pipeline import folder, read_jsonl, write_json


def gated_value(scores, task, short_side, min_side, gate):
    """(value, threshold head) for one video, task and gate setting; (None, None) if undefined."""
    if gate and short_side is not None and short_side < min_side and "audio" in scores:
        if task == "ai":
            return scores["audio"], "audio"
        if task in ("visual", "mismatch"):
            return None, None
    value = scores.get(task)
    return value, task


def has_gate(entry):
    """Only detectors with separate visual and audio evidence can fall back to audio (M3-R and
    BL-Ens-AV); single-score detectors (BL-M/sup, SpeechForensics) report the same score for
    every task and are left ungated."""
    if entry.get("ensemble"):
        return True
    heads = set(((entry.get("describe") or {}).get("heads") or {}).values())
    return {"visual", "audio"} <= heads


def load_scores(cross, name):
    out = {}
    for path in sorted((folder(cross, name) / "scores").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data["key"]] = data
    return out


def coverage(rows, media_all, common):
    by = {}
    errors = {m["sample_id"] for m in media_all if "error" in m}
    for r in rows:
        c = by.setdefault(r["category"], Counter())
        c["sampled"] += 1
        if r["sample_id"] in errors:
            c["media_error"] += 1
        elif r["sample_id"] not in common:
            c["not_scored_by_all"] += 1
        else:
            c["evaluated"] += 1
    return {k: dict(v) for k, v in sorted(by.items())}


def evaluate_dataset(cross, name, roster):
    proto = cross["protocols"]
    min_side = int(proto.get("gate_min_side", 360))
    reps, seed = int(proto.get("bootstrap", 1000)), int(cross["seed"])
    fprs = tuple(proto.get("fpr_targets") or (0.01, 0.05))
    rows = read_jsonl(folder(cross, name) / "manifest.jsonl")
    media_all = read_jsonl(folder(cross, name) / "media.jsonl")
    media = {m["sample_id"]: m for m in media_all if "error" not in m}
    scores = load_scores(cross, name)
    primary = [k for k in scores if not (roster.get(k) or {}).get("member_only")]
    common = set(media)
    for k in primary:
        common &= {sid for sid, v in scores[k]["scores"].items() if "base" in v}
    rows_c = [r for r in rows if r["sample_id"] in common]
    out = {"display": ds.DISPLAY.get(name, name), "tasks": list(ds.TASKS[name]), "n_common": len(rows_c),
           "coverage": coverage(rows, media_all, common), "models": {},
           "gated_share": float(np.mean([media[r["sample_id"]]["short_side"] < min_side for r in rows_c])) if rows_c else None,
           "short_side": [media[r["sample_id"]]["short_side"] for r in rows_c],
           "lead_silence_s": {"real": [media[r["sample_id"]]["lead_silence_s"] for r in rows_c if not r["label"]],
                              "fake": [media[r["sample_id"]]["lead_silence_s"] for r in rows_c if r["label"]]}}
    if not rows_c:
        return out
    labels_ai = [r["label"] for r in rows_c]
    # Shortcut probes: one raw feature as the score (AVH-Align's silence check, duration, resolution).
    out["shortcut_probes"] = {
        feat: M.auc(labels_ai, [float(media[r["sample_id"]].get(feat) or 0.0) for r in rows_c])
        for feat in ("lead_silence_s", "duration_s", "short_side")
    }
    curves, hist = {}, {}
    for key in primary:
        entry = roster.get(key) or {}
        thresholds = entry.get("thresholds") or {}
        data = scores[key]["scores"]
        gates = (False, True) if has_gate(entry) else (False,)
        model_out = {"errors": len(scores[key].get("errors") or {}),
                     "missing_features": scores[key].get("missing_features", 0), "results": {}}
        for condition in scores[key]["conditions"]:
            for gate in gates:
                tag = f"{condition}|gate_{'on' if gate else 'off'}"
                block = {}
                for task in ds.TASKS[name]:
                    picked, values, margins = [], [], []
                    for r in rows_c:
                        s = data[r["sample_id"]].get(condition)
                        if s is None:
                            continue
                        value, head = gated_value(s, task, media[r["sample_id"]]["short_side"], min_side, gate)
                        if value is None:
                            continue
                        picked.append(r)
                        values.append(value)
                        t = thresholds.get(head)
                        margins.append(None if t is None else value - t)
                    if not picked:
                        continue
                    result = M.task_block(picked, values, task, None, fprs, reps, seed)
                    if result is None:
                        continue
                    if all(m is not None for m in margins):
                        result["transfer"] = M.rates([M.target(r, task) for r in picked], margins, 0.0)
                    if task == "ai" and condition == "base":
                        result["by_category"] = M.group_auc(picked, values, "category")
                        result["by_method"] = M.group_auc(picked, values, "method")
                        if name == "mavosdd":
                            result["by_language"] = M.group_auc(picked, values, "language")
                            result["by_subset"] = M.subset_auc(picked, values, ds.MAVOS_TEST_SETS)
                        if not gate or len(gates) == 1:
                            fpr, tpr = M.roc_points([M.target(r, task) for r in picked], values)
                            curves[key] = {"fpr": fpr, "tpr": tpr, "auc": result["auc"]}
                            hist[key] = {"real": [v for r, v in zip(picked, values) if not r["label"]],
                                         "fake": [v for r, v in zip(picked, values) if r["label"]],
                                         "threshold": thresholds.get("ai")}
                    block[task] = result
                model_out["results"][tag] = block
        out["models"][key] = model_out
    out["roc"] = curves
    out["hist"] = hist
    out["paired"] = paired_deltas(cross, rows_c, scores, primary, media, min_side, reps, seed, roster)
    return out


def paired_deltas(cross, rows, scores, keys, media, min_side, reps, seed, roster):
    """Reference model (each seed, gate off and on) minus every other detector, AI? task, base.
    A gate-capable other detector uses the same gate setting as the reference."""
    reference = cross["models"].get("reference", "trihead_v5")
    labels = [r["label"] for r in rows]

    def column(key, gate):
        out = []
        for r in rows:
            value, _ = gated_value(scores[key]["scores"][r["sample_id"]]["base"], "ai",
                                   media[r["sample_id"]]["short_side"], min_side, gate)
            out.append(value)
        return out

    result = {}
    for ref in [k for k in keys if k.rsplit("_seed", 1)[0] == reference]:
        for gate in (False, True):
            ref_values = column(ref, gate)
            if any(v is None for v in ref_values):
                continue
            ref_seed = ref.rsplit("_seed", 1)[1]
            for other in keys:
                # same seed only (a seed-42 M3-R against seed-43 baselines says nothing); seedless = published
                if other == ref or ("_seed" in other and other.rsplit("_seed", 1)[1] != ref_seed):
                    continue
                other_values = column(other, gate and has_gate(roster.get(other) or {}))
                if any(v is None for v in other_values):
                    continue
                result[f"{ref}|gate_{'on' if gate else 'off'}|{other}"] = M.paired(
                    rows, labels, ref_values, other_values, reps, seed)
    return result


def table_rows(results, roster):
    out = []
    for name, d in results["datasets"].items():
        for key, m in d.get("models", {}).items():
            for tag, block in m["results"].items():
                condition, gate = tag.split("|")
                for task, r in block.items():
                    cal = (r.get("calibrated") or {}).get("fpr_0.05") or {}
                    tr = r.get("transfer") or {}
                    o1, o5 = (r["oracle"].get("fpr_0.01") or {}), (r["oracle"].get("fpr_0.05") or {})
                    out.append({
                        "dataset": name, "model": key, "display": (roster.get(key) or {}).get("describe", {}).get("display", key),
                        "condition": condition, "gate": gate, "task": task, "n": r["n"], "n_fake": r["n_fake"],
                        "auc": r["auc"], "auc_lo": (r["auc_ci"] or [None, None])[0], "auc_hi": (r["auc_ci"] or [None, None])[1],
                        "ap": r["ap"], "eer": r["eer"], "tpr_at_1fpr": o1.get("tpr"), "tpr_at_5fpr": o5.get("tpr"),
                        "transfer_fpr": tr.get("fpr"), "transfer_tpr": tr.get("tpr"),
                        "calibrated_fpr": cal.get("fpr"), "calibrated_tpr": cal.get("tpr"),
                    })
    return out


def step_evaluate(cross, only=None):
    results_dir = Path(cross["work"]) / "results"
    roster = json.loads((results_dir / "roster.json").read_text(encoding="utf-8"))
    results = {"protocols": cross["protocols"], "seed": cross["seed"], "datasets": {}}
    # every dataset with scores in the work folder (the report session merges several sessions and
    # does not need the videos attached)
    names = [n for n in (cross.get("datasets") or {}) if (not only or n in only) and (folder(cross, n) / "scores").is_dir()]
    for name in names:
        results["datasets"][name] = evaluate_dataset(cross, name, roster)
        print(f"evaluated {name}: {results['datasets'][name]['n_common']} common videos", flush=True)
    write_json(results_dir / "metrics.json", results)
    rows = table_rows(results, roster)
    if rows:
        with open(results_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return results
