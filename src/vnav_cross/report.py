"""Report: figures, Markdown tables (English) and the W&B report run.

Writes <work>/results/report.md, figures/*.png and tables/*.csv; logs all of them to W&B
(job_type "report"). Numbers come only from metrics.json / roster.json / audit.json and from
configs/published_results.yaml (kept in its own section, never mixed with our protocol).
"""

import csv
import json
from pathlib import Path

import yaml

from vnav_cross import figures as F
from vnav_cross.datasets import DISPLAY
from vnav_cross.tracking import open_run

STREAM_LABEL = {"trace": "lip-head trace", "sync_rel": "sync (relative)", "sync_abs": "sync (absolute)",
                "phi": "AVH-Align phi channel", "audio": "audio (XLS-R)"}


def f3(v):
    return "-" if v is None else f"{v:.3f}"


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_csv(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def block(d, key, tag, task="ai"):
    return ((d["models"].get(key) or {}).get("results", {}).get(tag) or {}).get(task)


def name_of(roster, key):
    return (roster.get(key) or {}).get("describe", {}).get("display", key)


def sorted_keys(keys):
    def rank(k):
        n = k.rsplit("_seed", 1)[0]
        return (F.ORDER.index(n) if n in F.ORDER else 99, k)

    return sorted(keys, key=rank)


# ---------------------------------------------------------------- sections
def section_detectors(roster, figs, out):
    rows, bars = [], []
    for key in sorted_keys(k for k, e in roster.items() if not e.get("member_only")):
        d = roster[key].get("describe", {})
        heads = ", ".join(sorted(set((d.get("heads") or {}).values()))) or ("ai (late fusion)" if "members" in d else "-")
        th = roster[key].get("thresholds") or {}
        rows.append([d.get("display", key), d.get("role", ""), ", ".join(d.get("needs") or []) or "-", heads,
                     d.get("pool") or "-", f"{d['parameters']:,}" if d.get("parameters") else "-",
                     d.get("train_data", "-"), f3(th.get("ai")) if th else "none",
                     f3((roster[key].get("in_domain") or {}).get("clean"))])
        if d.get("modules"):
            bars.append((d.get("display", key), [(m["module"], m["parameters"]) for m in d["modules"]]))
    if bars:
        figs.append(F.stacked_params(bars, out / "figures/parameters.png", "Detector size by module"))
    write_csv(out / "tables/detectors.csv", ["detector", "role", "inputs", "outputs", "pooling", "parameters",
                                             "training data", "AI? threshold (VN val, 5% FPR)", "VN-AV-DF test AUC"], rows)
    text = ["## 2. Detectors", "",
            "Inputs: `av` = AV-HuBERT Large audio and visual features (25 fps, 1024-d, frozen), `mouth` = 96 px "
            "mouth crops + 20 lip landmarks + head pose, `ssl` = frozen XLS-R 300M hidden states (5 layer groups). "
            "Frozen encoders are not counted in the parameters. The threshold column is the AI? threshold at 5% FPR "
            "on VN-AV-DF validation reals, used unchanged in the transfer test.", "",
            md_table(["Detector", "Role", "Inputs", "Outputs", "Pooling", "Parameters", "Training data",
                      "AI? threshold", "VN-AV-DF test AUC"], rows), ""]
    if bars:
        text += ["![parameters](figures/parameters.png)", ""]
    return text


def section_data(results, out):
    rows = []
    for name, d in results["datasets"].items():
        cov = d["coverage"]
        for cat, c in cov.items():
            rows.append([DISPLAY.get(name, name), cat, c.get("sampled", 0), c.get("media_error", 0),
                         c.get("not_scored_by_all", 0), c.get("evaluated", 0)])
    write_csv(out / "tables/coverage.csv", ["dataset", "class", "sampled", "media error", "features / scores missing", "evaluated"], rows)
    return ["## 1. Data and coverage", "",
            "Each dataset is sampled per class with a fixed seed (configs/crossdata.yaml); every detector is "
            "evaluated on the same videos (those that every detector could score). Videos lost at each stage "
            "are listed, because dropping hard videos (no face, no audio) would inflate the AUC.", "",
            md_table(["Dataset", "Class", "Sampled", "Media error", "Not scored by all", "Evaluated"], rows), ""]


def section_main(results, roster, figs, out):
    names = list(results["datasets"])
    keys = sorted_keys({k for d in results["datasets"].values() for k in d.get("models", {})})
    text = ["## 3. Zero-shot AUC (AI? task)", "",
            "Base condition (normalized video, no trimming). 95% CI: bootstrap over subjects. "
            "M3-R is shown with its shipped 360p resolution gate (gate on) and without it (gate off).", ""]
    header = ["Detector", "VN-AV-DF (in-domain)"] + [DISPLAY.get(n, n) for n in names]
    rows, matrix, labels = [], [], []
    for key in keys:
        tags = ["base|gate_off", "base|gate_on"] if any(
            "base|gate_on" in (d["models"].get(key) or {}).get("results", {}) for d in results["datasets"].values()) else ["base|gate_off"]
        for tag in tags:
            label = name_of(roster, key) + (" [gate on]" if tag.endswith("on") else "")
            line, values = [label, f3((roster.get(key) or {}).get("in_domain", {}).get("clean"))], [
                (roster.get(key) or {}).get("in_domain", {}).get("clean")]
            for n in names:
                b = block(results["datasets"][n], key, tag)
                line.append("-" if b is None else f"{b['auc']:.3f} [{b['auc_ci'][0]:.3f}, {b['auc_ci'][1]:.3f}]"
                            if b.get("auc_ci") else f3(b["auc"]))
                values.append(None if b is None else b["auc"])
            rows.append(line)
            matrix.append(values)
            labels.append(label)
    write_csv(out / "tables/auc_ai.csv", header, rows)
    figs.append(F.heatmap(matrix, labels, ["VN-AV-DF"] + [DISPLAY.get(n, n) for n in names],
                          out / "figures/auc_heatmap_ai.png", "AI? AUC: in-domain vs zero-shot"))
    text += [md_table(header, rows), "", "![auc](figures/auc_heatmap_ai.png)", ""]
    for task in ("visual", "audio"):
        t_rows, t_matrix, t_labels = [], [], []
        cols = [n for n in names if task in results["datasets"][n]["tasks"]]
        if not cols:
            continue
        for key in keys:
            values = [None if block(results["datasets"][n], key, "base|gate_off", task) is None
                      else block(results["datasets"][n], key, "base|gate_off", task)["auc"] for n in cols]
            if all(v is None for v in values):
                continue
            t_rows.append([name_of(roster, key)] + [f3(v) for v in values])
            t_matrix.append(values)
            t_labels.append(name_of(roster, key))
        if t_rows:
            figs.append(F.heatmap(t_matrix, t_labels, [DISPLAY.get(n, n) for n in cols],
                                  out / f"figures/auc_heatmap_{task}.png", f"{task.capitalize()} task AUC (zero-shot)"))
            text += [f"### {task.capitalize()} task", "", md_table(["Detector"] + [DISPLAY.get(n, n) for n in cols], t_rows), "",
                     f"![{task}](figures/auc_heatmap_{task}.png)", ""]
    return text


def seeds_of(keys):
    """{model name: [keys of each seed]} for models with a seed suffix."""
    out = {}
    for k in sorted_keys(keys):
        if "_seed" in k:
            out.setdefault(k.rsplit("_seed", 1)[0], []).append(k)
    return out


def section_seeds(results, roster, figs, out):
    """Mean over seeds with the min-max range: the seed-to-seed spread is the first thing to beat."""
    names = list(results["datasets"])
    groups = [DISPLAY.get(n, n) for n in names]
    by_model = seeds_of({k for d in results["datasets"].values() for k in d.get("models", {})})
    rows, values, errors, series = [], {}, {}, []
    for model, keys in by_model.items():
        if len(keys) < 2:
            continue
        gate_tags = ["base|gate_off"]
        if any("base|gate_on" in (d["models"].get(keys[0]) or {}).get("results", {}) for d in results["datasets"].values()):
            gate_tags.append("base|gate_on")
        for tag in gate_tags:
            label = name_of(roster, keys[0]).rsplit(" (s", 1)[0] + (" [gate on]" if tag.endswith("on") else "")
            series_key = model if tag.endswith("off") else f"{model}|on"
            line = [label, ", ".join(k.rsplit("_seed", 1)[1] for k in keys),
                    "/".join(f3((roster.get(k) or {}).get("in_domain", {}).get("clean")) for k in keys)]
            values[series_key], errors[series_key] = {}, {}
            for n, g in zip(names, groups, strict=True):
                aucs = [b["auc"] for k in keys if (b := block(results["datasets"][n], k, tag)) is not None]
                if len(aucs) == len(keys):
                    mean = float(sum(aucs) / len(aucs))
                    values[series_key][g], errors[series_key][g] = mean, [min(aucs), max(aucs)]
                    line.append(f"{mean:.3f} ({' / '.join(f'{a:.3f}' for a in aucs)})")
                else:
                    line.append("-")
            rows.append(line)
            if tag.endswith("off"):
                series.append((series_key, label))
    if not rows:
        return []
    figs.append(F.grouped_bars(groups, series, values, out / "figures/auc_seed_mean.png",
                               "AI? AUC, mean over seeds (bars: min-max of the seeds)", "AUC",
                               ref=0.5, ref_label="chance", errors=errors))
    header = ["Detector", "Seeds", "VN-AV-DF (per seed)"] + [f"{g} mean (per seed)" for g in groups]
    write_csv(out / "tables/auc_seed_mean.csv", header, rows)
    return ["## 3b. Two seeds", "",
            "Every detector of ours was trained twice on VN-AV-DF (seed 42 in v005, seed 43 in v006) and both "
            "checkpoints are scored on the same videos. A gap between two detectors smaller than their "
            "seed-to-seed spread is not a difference.", "",
            md_table(header, rows), "", "![seeds](figures/auc_seed_mean.png)", ""]


def section_pairing(results, roster):
    """Retrained ablation: M3-R vs M3-R without the clean <-> re-encoded pairing, same seed, same videos."""
    rows = []
    for n, d in results["datasets"].items():
        for key in sorted_keys(d.get("models", {})):
            if key.rsplit("_seed", 1)[0] != "trihead_v5" or "_seed" not in key:
                continue
            seed = key.rsplit("_seed", 1)[1]
            other = f"trihead_v5_no_pair_seed{seed}"
            a, b = block(d, key, "base|gate_off"), block(d, other, "base|gate_off")
            p = (d.get("paired") or {}).get(f"{key}|gate_off|{other}")
            if a and b:
                rows.append([DISPLAY.get(n, n), seed, f3(a["auc"]), f3(b["auc"]),
                             "-" if p is None else f"{p['delta']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"])
    if not rows:
        return []
    return ["### Retrained ablation: clean <-> re-encoded pairing", "",
            "`trihead_v5_no_pair` is M3-R trained with random re-encoded copies instead of paired ones (no "
            "consistency loss); in-domain this is gate R6 at 480p (+0.004 at seed 42, -0.001 at seed 43). Here the same "
            "comparison on external data, gate off.", "",
            md_table(["Dataset", "Seed", "M3-R", "w/o pairing", "Delta [95% CI]"], rows), ""]



def section_per_dataset(results, roster, figs, out):
    text = ["## 4. Per dataset: ROC, classes and score distributions", ""]
    for name, d in results["datasets"].items():
        if not d.get("models"):
            continue
        label = DISPLAY.get(name, name)
        keys = sorted_keys(d["roc"])
        curves = [(k, name_of(roster, k), d["roc"][k]["fpr"], d["roc"][k]["tpr"], d["roc"][k]["auc"]) for k in keys]
        figs.append(F.roc(curves, out / f"figures/roc_{name}.png", f"{label}: ROC (AI?, gate off)"))
        cats = sorted({c for k in keys for c in ((block(d, k, "base|gate_off") or {}).get("by_category") or {})})
        values = {k: {c: ((block(d, k, "base|gate_off") or {}).get("by_category") or {}).get(c, {}).get("auc") for c in cats} for k in keys}
        figs.append(F.grouped_bars(cats, [(k, name_of(roster, k)) for k in keys], values,
                                   out / f"figures/classes_{name}.png", f"{label}: AUC per fake class vs reals", "AUC",
                                   ref=0.5, ref_label="chance"))
        rows = [[name_of(roster, k)] + [f3(values[k][c]) for c in cats] for k in keys]
        text += [f"### {label}", "", f"Common videos: {d['n_common']}; share below the 360p gate: {pct(d.get('gated_share'))}.", "",
                 md_table(["Detector"] + cats, rows), "",
                 f"![roc](figures/roc_{name}.png) ![classes](figures/classes_{name}.png)", ""]
        methods = sorted({m for k in keys for m in ((block(d, k, "base|gate_off") or {}).get("by_method") or {})})
        if len(methods) > 1:
            mrows = [[name_of(roster, k)] + [f3(((block(d, k, "base|gate_off") or {}).get("by_method") or {}).get(m, {}).get("auc")) for m in methods] for k in keys]
            text += ["Per generator / method:", "", md_table(["Detector"] + methods, mrows), ""]
        if name == "mavosdd":
            subsets = ["in_domain", "open_model", "open_language", "open_full"]
            sv = {k: {s: ((block(d, k, "base|gate_off") or {}).get("by_subset") or {}).get(s, {}).get("auc") for s in subsets} for k in keys}
            figs.append(F.grouped_bars(subsets, [(k, name_of(roster, k)) for k in keys], sv,
                                       out / "figures/mavos_subsets.png", "MAVOS-DD test sets", "AUC", ref=0.5, ref_label="chance"))
            text += ["MAVOS-DD test sets (paper definitions):", "", md_table(["Detector"] + subsets, [[name_of(roster, k)] + [f3(sv[k][s]) for s in subsets] for k in keys]), "",
                     "![mavos](figures/mavos_subsets.png)", ""]
        for k in keys:
            if k.rsplit("_seed", 1)[0] in ("trihead_v5", "avh_align_plus") and k in d["hist"]:
                h = d["hist"][k]
                figs.append(F.histograms([("real", h["real"], "#0072B2"), ("fake", h["fake"], "#D55E00")],
                                         out / f"figures/scores_{name}_{k}.png", f"{label}: {name_of(roster, k)} AI? scores",
                                         threshold=h.get("threshold")))
                text += [f"![scores](figures/scores_{name}_{k}.png)", ""]
    return text


def section_operating(results, roster, figs, out):
    names = list(results["datasets"])
    keys = sorted_keys({k for d in results["datasets"].values() for k in d.get("models", {})})
    groups = [DISPLAY.get(n, n) for n in names]
    transfer = {k: {} for k in keys}
    calibrated = {k: {} for k in keys}
    rows = []
    for n, g in zip(names, groups):
        for k in keys:
            b = block(results["datasets"][n], k, "base|gate_on") or block(results["datasets"][n], k, "base|gate_off")
            if b is None:
                continue
            tr, cal = b.get("transfer") or {}, (b.get("calibrated") or {}).get("fpr_0.05") or {}
            transfer[k][g], calibrated[k][g] = tr.get("fpr"), cal.get("fpr")
            o5 = b["oracle"].get("fpr_0.05") or {}
            rows.append([g, name_of(roster, k), pct(tr.get("fpr")), pct(tr.get("tpr")), pct(cal.get("fpr")),
                         pct(cal.get("tpr")), pct(o5.get("tpr")), f3(b.get("eer"))])
    series = [(k, name_of(roster, k)) for k in keys if any(v is not None for v in transfer[k].values())]
    if series:
        top = max([v for k, _ in series for v in transfer[k].values() if v is not None] + [0.1])
        figs.append(F.grouped_bars(groups, series, transfer, out / "figures/transfer_fpr.png",
                                   "False alarms on target reals at the VN-AV-DF threshold", "FPR on reals",
                                   ylim=(0, min(1.0, top * 1.15)), ref=0.05, ref_label="nominal 5%"))
    cal_series = [(k, name_of(roster, k)) for k in keys if any(v is not None for v in calibrated[k].values())]
    if cal_series:
        figs.append(F.grouped_bars(groups, cal_series, calibrated, out / "figures/calibrated_fpr.png",
                                   "FPR after re-fitting the threshold on target calibration reals", "FPR on reals",
                                   ylim=(0, 0.2), ref=0.05, ref_label="target 5%"))
    write_csv(out / "tables/operating_points.csv", ["dataset", "detector", "transfer FPR", "transfer TPR",
                                                    "calibrated FPR", "calibrated TPR", "oracle TPR@5%FPR", "EER"], rows)
    return ["## 5. Operating points: does the VN-AV-DF threshold transfer?", "",
            "Transfer = VN-AV-DF threshold (5% FPR on its validation reals) applied unchanged. Calibrated = threshold "
            "re-fitted at 5% FPR on the reals of a subject-disjoint calibration part (30% of subjects) of the target, "
            "rates measured on the other 70%. Oracle = threshold set on the evaluated reals themselves (upper bound). "
            "Gate on for detectors that have it.", "",
            md_table(["Dataset", "Detector", "Transfer FPR", "Transfer TPR", "Calibrated FPR", "Calibrated TPR",
                      "Oracle TPR@5%FPR", "EER"], rows), "",
            "![transfer](figures/transfer_fpr.png) ![calibrated](figures/calibrated_fpr.png)", ""]


def section_paired(results, roster, figs, out):
    rows, forest = [], []
    for n, d in results["datasets"].items():
        for tag, p in sorted((d.get("paired") or {}).items()):
            if p is None:
                continue
            ref, gate, other = tag.split("|")
            label = f"{DISPLAY.get(n, n)} | {gate.replace('_', ' ')} | vs {name_of(roster, other)}"
            rows.append([DISPLAY.get(n, n), name_of(roster, ref), gate.replace("gate_", ""), name_of(roster, other),
                         f"{p['delta']:+.3f}", f"[{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]", f"{p['p_le_0']:.3f}"])
            forest.append((label, other, p["delta"], p["ci"][0], p["ci"][1]))
    if forest:
        figs.append(F.forest(forest, out / "figures/paired_deltas.png", "M3-R minus other detectors (AI? AUC, same videos)",
                             "AUC(M3-R) - AUC(other), 95% subject-bootstrap CI"))
    write_csv(out / "tables/paired_deltas.csv", ["dataset", "reference", "gate", "other", "delta", "95% CI", "P(delta<=0)"], rows)
    return ["## 6. Paired comparison with M3-R", "",
            "Difference of AUC on the same videos; the CI resamples subjects, so a detector that wins on a few "
            "identities only gets a wide interval. These are descriptive comparisons, not pre-registered gates.", "",
            md_table(["Dataset", "Reference", "Gate", "Other", "Delta AUC", "95% CI", "P(delta<=0)"], rows), "",
            "![paired](figures/paired_deltas.png)" if forest else "", ""]


def section_shortcuts(results, roster, figs, out):
    rows, dumb = [], []
    for n, d in results["datasets"].items():
        probes = d.get("shortcut_probes") or {}
        rows.append([DISPLAY.get(n, n)] + [f3(probes.get(f)) for f in ("lead_silence_s", "duration_s", "short_side")])
        for k in sorted_keys(d.get("models", {})):
            a, b = block(d, k, "base|gate_off"), block(d, k, "trim|gate_off")
            if a and b:
                dumb.append((f"{DISPLAY.get(n, n)}: {name_of(roster, k)}", k, a["auc"], b["auc"]))
    if dumb:
        figs.append(F.dumbbell(dumb, out / "figures/trim.png", "Leading silence removed (AVH-Align check)"))
    return ["## 7. Shortcut checks", "",
            "Single-feature probes use one raw measurement as the score (AUC far from 0.5 = the dataset can be "
            "separated without looking at faces or voices; Smeu et al. report a leading-silence classifier at "
            "98.4 AUC on FakeAVCeleb). Trim = the leading silence cut from every frame stream before scoring "
            "(feature slicing, not re-extraction).", "",
            md_table(["Dataset", "Leading silence", "Duration", "Short side"], rows), "",
            "![trim](figures/trim.png)" if dumb else "", ""]


def section_gate(results, roster, figs, out):
    rows, dumb = [], []
    per = {}
    for n, d in results["datasets"].items():
        per[DISPLAY.get(n, n)] = d.get("short_side") or [0]
        for k in sorted_keys(d.get("models", {})):
            off, on = block(d, k, "base|gate_off"), block(d, k, "base|gate_on")
            if off and on:
                rows.append([DISPLAY.get(n, n), name_of(roster, k), pct(d.get("gated_share")), f3(off["auc"]), f3(on["auc"])])
                dumb.append((f"{DISPLAY.get(n, n)}: {name_of(roster, k)}", k, off["auc"], on["auc"]))
    figs.append(F.resolution(per, out / "figures/resolution.png"))
    if dumb:
        figs.append(F.dumbbell(dumb, out / "figures/gate.png", "Resolution gate off vs on", "gate off", "gate on"))
    return ["## 8. Resolution gate", "",
            "M3-R skips the visual evidence below 360p (its lowest training level) and decides from the audio head. "
            "Many public datasets are face crops below that size, so the gate decides how much of M3-R is tested.", "",
            md_table(["Dataset", "Detector", "Videos gated", "AUC gate off", "AUC gate on"], rows), "",
            "![resolution](figures/resolution.png) " + ("![gate](figures/gate.png)" if dumb else ""), ""]


def section_ablation(results, roster, figs, out):
    names = list(results["datasets"])
    groups = [DISPLAY.get(n, n) for n in names]
    text = ["## 9. Test-time ablation of M3-R", "",
            "One branch output is set to zero before the heads, with the trained weights unchanged (gate off). "
            "The drop shows how much the trained heads rely on that branch on each dataset; it is not a "
            "retrained ablation (those are run in vn-av-df-forensics, e.g. `trihead_v5_no_pair`, `trihead_v5_clean`).", ""]
    for key in sorted_keys({k for d in results["datasets"].values() for k in d.get("models", {})}):
        if key.rsplit("_seed", 1)[0] != "trihead_v5":
            continue
        values, rows = {}, []
        streams = []
        for n, g in zip(names, groups):
            d = results["datasets"][n]
            base = block(d, key, "base|gate_off")
            if base is None:
                continue
            for tag, res in (d["models"][key]["results"]).items():
                if not tag.startswith("knockout:") or not tag.endswith("gate_off") or "ai" not in res:
                    continue
                stream = tag.split(":", 1)[1].split("|")[0]
                if stream not in streams:
                    streams.append(stream)
                values.setdefault(stream, {})[g] = res["ai"]["auc"] - base["auc"]
        if not streams:
            continue
        for s in streams:
            rows.append([STREAM_LABEL.get(s, s)] + [f"{values[s][g]:+.3f}" if g in values[s] else "-" for g in groups])
        series = [(s, STREAM_LABEL.get(s, s), F.PALETTE[i % len(F.PALETTE)]) for i, s in enumerate(streams)]
        every = [v for s in streams for v in values[s].values()]
        low, high = min(every + [-0.02]), max(every + [0.02])
        pad = 0.1 * (high - low)
        figs.append(F.grouped_bars(groups, series, values, out / f"figures/knockout_{key}.png",
                                   f"{name_of(roster, key)}: AUC change when one branch is zeroed", "delta AUC (AI?)",
                                   ylim=(low - pad, high + pad), ref=0.0, ref_label="no change"))
        text += [f"### {name_of(roster, key)}", "", md_table(["Branch zeroed"] + groups, rows), "",
                 f"![knockout](figures/knockout_{key}.png)", ""]
    return text


def section_published(path):
    entries = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else []
    text = ["## 10. Numbers reported by other papers (context only)", "",
            "Copied verbatim from the cited tables. Their test subsets, preprocessing and training data differ from "
            "ours, so these are not a ranking against Section 3.", ""]
    for e in entries:
        cols = e["columns"]
        text += [f"{e['source']}. Metric: {e['metric']}. Protocol: {e['protocol']}", "",
                 md_table(["Method", "Training"] + cols, [[r["method"], r["train"]] + [r.get(c, "-") for c in cols] for r in e["rows"]]), ""]
    return text


CAVEATS = [
    "## 11. Caveats", "",
    "- AV-HuBERT (all audio-visual detectors here) was pre-trained on LRS3 and VoxCeleb2; FakeAVCeleb, "
    "LAV-DF and AV-Deepfake1M are built from VoxCeleb2 videos, so the encoder may have seen their real speakers "
    "(self-supervised, no labels).",
    "- The released AVH-Align checkpoint, which also initializes the phi network of BL-M, M2 and M3-R, was trained "
    "on AV-Deepfake1M train reals; AV-Deepfake1M validation shares that source distribution.",
    "- Our detectors were trained on Vietnamese speech only; the external sets are English (and MAVOS-DD "
    "multilingual), so every external number is a cross-language transfer.",
    "- MAVOS-DD fakes keep the real speech: the audio head cannot help there, and the gate-on M3-R score falls "
    "back to audio for low-resolution videos.",
    "- Trim uses feature slicing; AV-HuBERT context near the cut differs slightly from re-encoding a trimmed file.",
    "- Sampling: up to N videos per class (configs/crossdata.yaml), not the full test sets; the sampling table "
    "in sampling.json gives full and sampled counts.", "",
]


def step_report(cross, results=None):
    out = Path(cross["work"]) / "results"
    results = results or json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    roster = json.loads((out / "roster.json").read_text(encoding="utf-8"))
    figs = []
    text = ["# Cross-dataset evaluation of VN-AV-DF detectors", "",
            "Detectors trained on VN-AV-DF (Vietnamese) scored zero-shot on public English / multilingual "
            "audio-visual deepfake datasets, next to published detectors run with their authors' weights.", ""]
    text += section_data(results, out)
    text += section_detectors(roster, figs, out)
    text += section_main(results, roster, figs, out)
    text += section_seeds(results, roster, figs, out)
    text += section_per_dataset(results, roster, figs, out)
    text += section_operating(results, roster, figs, out)
    text += section_paired(results, roster, figs, out)
    text += section_shortcuts(results, roster, figs, out)
    text += section_gate(results, roster, figs, out)
    text += section_ablation(results, roster, figs, out)
    text += section_pairing(results, roster)
    text += section_published(cross["published_results"])
    text += CAVEATS
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    run = open_run(cross, "report", "report", config={"datasets": list(results["datasets"]), "protocols": cross["protocols"]})
    run.images(figs)
    for table in sorted((out / "tables").glob("*.csv")):
        with open(table, encoding="utf-8") as f:
            data = list(csv.reader(f))
        if len(data) > 1:
            run.table(f"tables/{table.stem}", data[0], data[1:])
    for name, d in results["datasets"].items():
        for key in d.get("models", {}):
            b = block(d, key, "base|gate_on") or block(d, key, "base|gate_off")
            if b:
                run.summary({f"{name}/{key}/auc_ai": b["auc"], f"{name}/{key}/ap_ai": b["ap"]})
    run.artifact("crossdata-results", "results", [out / "report.md", out / "metrics.json", out / "metrics.csv",
                                                  out / "roster.json", out / "figures", out / "tables"])
    run.finish()
    print(f"report: {out / 'report.md'} ({len(figs)} figures)", flush=True)
    return out / "report.md"


# ---------------------------------------------------------------- audit report
def step_audit_report(cross, audit, out):
    out = Path(out)
    figs = []
    short = audit.get("shortcuts") or {}
    feats = sorted({f for s in short.values() for f in (s.get("overall") or {})})
    if feats:
        series = [(p, p.replace("vn-av-df-data-", ""), F.PALETTE[i % len(F.PALETTE)]) for i, p in enumerate(short)]
        values = {p: {f: short[p]["overall"].get(f) for f in feats} for p in short}
        figs.append(F.grouped_bars(feats, series, values, out / "figures/audit_shortcuts.png",
                                   "VN-AV-DF: single-feature AUC, AI vs not-AI (0.5 = no shortcut)", "AUC",
                                   ylim=(0.25, 0.75), ref=0.5, ref_label="chance"))
    over = audit["overlap"]
    reuse = audit["test_reuse"]
    lines = ["# VN-AV-DF leakage audit", "",
             f"Rows: {audit['rows']} (parts: {', '.join(audit['splits']['files'])}). Speakers per split: "
             + ", ".join(f"{k} {v}" for k, v in audit["splits"]["speakers"].items()) + ".", "",
             "## Overlap between splits", "",
             md_table(["Key", "Distinct values", "Shared across splits", "By split pair"],
                      [[k, v["values"], v["shared"], v["by_pair"] or "-"] for k, v in over.items()]), "",
             f"Borrowed-audio clips (sham / donor / AI voice reference from another video): {audit['donor_audio']['borrowed_audio_clips']}; "
             f"crossing splits: {sum(audit['donor_audio']['crossing_splits'].values())}.", "",
             f"Transcripts: {audit['transcripts']['sentences']} distinct, shared across splits: {audit['transcripts']['shared']}.", "",
             "## Test reuse", "",
             f"Logged test uses: {reuse['entries']} in rounds {', '.join(reuse['rounds'])} ({reuse['sessions']} sessions); "
             f"trained checkpoints: {reuse['checkpoints']}, tested more than once: {reuse['checkpoints_tested_more_than_once']} "
             "(training-free SpeechForensics excluded: it has no weights). Rounds without a test-log.json "
             "(the log starts in v005) are not counted, and the log cannot show design decisions taken after "
             "looking at test numbers; the external datasets of report.md are the answer to that.", "",
             md_table(["Round", "Checkpoint", "Session", "Time", "Use"], reuse["table"]), "",
             "## Single-feature shortcuts", "",
             md_table(["Part", "Feature", "AUC (all)", "Most extreme cell", "AUC (cell vs real)"],
                      [[p, f, f3((s.get("overall") or {}).get(f)), (s["most_extreme_cell"].get(f) or {}).get("cell"),
                        f3((s["most_extreme_cell"].get(f) or {}).get("auc"))] for p, s in short.items() for f in feats]), "",
             "Duration AUC per split (manifest): " + ", ".join(f"{k} {v:.3f}" for k, v in audit["duration_auc"].items()) + ".", ""]
    if figs:
        lines += ["![shortcuts](figures/audit_shortcuts.png)", ""]
    (out / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    run = open_run(cross, "audit", "audit-vnavdf", config={"manifests": audit["manifests"]})
    run.images(figs)
    run.table("audit/overlap", ["key", "values", "shared"], [[k, v["values"], v["shared"]] for k, v in over.items()])
    run.table("audit/test_reuse", ["round", "checkpoint", "session", "time", "use"], reuse["table"])
    run.summary({"leak/identity_or_source": len(audit["verdict"]["identity_or_source_leak"]),
                 "leak/donor_crossing": audit["verdict"]["donor_audio_crossing"],
                 "leak/repeated_test_use": audit["verdict"]["repeated_test_use"]})
    run.artifact("vnavdf-audit", "audit", [out / "audit.json", out / "audit.md"])
    run.finish()
    return out / "audit.md"
