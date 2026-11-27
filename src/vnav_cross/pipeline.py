"""Steps: manifest → prepare → extract → score (evaluate / report / audit live in their modules).

Work folder (cross["work"]):
    <dataset>/manifest.jsonl     sampled rows (datasets.py schema) + "part" (calibration / evaluation)
    <dataset>/sampling.json      full vs sampled counts per stratum
    <dataset>/media.jsonl        normalized clip, source resolution, leading silence, or the error
    <dataset>/clips/             normalized videos (25 fps, <= 720p)
    <dataset>/cache/<encoder>/   features, VN-AV-DF training layout
    <dataset>/scores/<key>.json  raw video scores per condition and task
    results/                     metrics.json, tables, figures, report.md
"""

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from vnav_cross import datasets, features, media, sampling, scoring
from vnav_cross.tracking import open_run

HERE = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- config
def load_config(path=None, **overrides):
    path = Path(path or HERE / "configs/crossdata.yaml")
    cross = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if value is not None:
            cross[key] = value
    cross["published_results"] = str(HERE / "configs/published_results.yaml")
    return cross


def active_datasets(cross, only=None):
    out = {}
    for name, spec in (cross.get("datasets") or {}).items():
        if only and name not in only:
            continue
        if spec and spec.get("root") and Path(spec["root"]).exists():
            out[name] = spec
        elif only:
            print(f"{name}: root not found ({spec.get('root') if spec else None}), skipped", flush=True)
    return out


def main_settings(cross):
    """settings.py of vn-av-df-forensics (encoders, weights, thresholds of the resolution gate)."""
    root = Path(cross.get("main_root") or os.environ.get("VNAVDF_ROOT") or "/kaggle/working/vn-av-df-forensics")
    loaded = sys.modules.get("settings")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == (root / "settings.py").resolve():
        return loaded
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))  # settings.py imports data_settings.py next to it
    spec = importlib.util.spec_from_file_location("settings", root / "settings.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["settings"] = module
    spec.loader.exec_module(module)
    return module


def main_cfg(cross, dataset, device=None):
    cfg = main_settings(cross).config()
    cfg["cache"] = str(Path(cross["work"]) / dataset / "cache")
    if device:
        cfg["device"] = device
        for spec in cfg["encoders"].values():
            if "device" in spec:
                spec["device"] = device
    cfg["wandb"] = cross.get("wandb")
    return cfg


def folder(cross, dataset):
    return Path(cross["work"]) / dataset


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- 1. manifest
def step_manifest(cross, only=None):
    for name, spec in active_datasets(cross, only).items():
        tick = time.time()
        full = datasets.index(name, spec)
        keys = list((spec.get("sample") or {}).get("strata") or ["category"])
        picked = sampling.sample(full, spec.get("sample") or {}, cross["seed"])
        picked = sampling.calibration_split(picked, cross["protocols"]["calibration_fraction"], cross["seed"])
        stats = sampling.sample_stats(full, picked, keys)
        write_jsonl(folder(cross, name) / "manifest.jsonl", picked)
        write_json(folder(cross, name) / "sampling.json", stats)
        print(f"{name}: {len(full)} videos indexed, {len(picked)} sampled "
              f"({stats['subjects_sampled']} subjects) in {time.time() - tick:.0f} s", flush=True)
        run = open_run(cross, "manifest", f"manifest-{name}", group=name,
                       config={"dataset": name, "spec": spec, "seed": cross["seed"]})
        run.table("sampling/strata", ["stratum", "full", "sampled"],
                  [[r["stratum"], r["full"], r["sampled"]] for r in stats["rows"]])
        run.summary({"videos_full": stats["full"], "videos_sampled": stats["sampled"],
                     "subjects_full": stats["subjects_full"], "subjects_sampled": stats["subjects_sampled"],
                     "calibration_rows": sum(r["part"] == "calibration" for r in picked)})
        run.finish()


# ---------------------------------------------------------------- 2. prepare (normalize, CPU)
def step_prepare(cross, only=None, workers=None):
    tau = float(cross["protocols"].get("silence_tau", 5e-4))
    for name in active_datasets(cross, only):
        rows = read_jsonl(folder(cross, name) / "manifest.jsonl")
        done = {m["sample_id"]: m for m in read_jsonl(folder(cross, name) / "media.jsonl") if "error" not in m}
        todo = [r for r in rows if r["sample_id"] not in done]
        clips = folder(cross, name) / "clips"
        clips.mkdir(parents=True, exist_ok=True)
        tick = time.time()
        # Run opened first: progress is visible (log + W&B) during the long normalization.
        run = open_run(cross, "prepare", f"prepare-{name}", group=name, config={"dataset": name, "tau": tau})
        print(f"{name}: normalizing {len(todo)} videos ({len(done)} already done)", flush=True)
        infos, failed = [], 0
        with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 2) as pool:
            for n, info in enumerate(pool.map(lambda r: media.prepare(r, clips, tau), todo), 1):
                infos.append(info)
                failed += "error" in info
                if n % 25 == 0 or n == len(todo):
                    spent = time.time() - tick
                    left = spent / n * (len(todo) - n)
                    print(f"  {name} prepare {n}/{len(todo)}, {failed} errors, {spent / 60:.1f} min, ~{left / 60:.0f} min left", flush=True)
                    run.log({"prepare/done": n, "prepare/errors": failed, "prepare/videos_per_min": n / max(spent, 1e-6) * 60})
        merged = {**done, **{i["sample_id"]: i for i in infos}}
        write_jsonl(folder(cross, name) / "media.jsonl", [merged[r["sample_id"]] for r in rows if r["sample_id"] in merged])
        errors = [i for i in infos if "error" in i]
        print(f"{name}: {len(todo)} normalized in {(time.time() - tick) / 60:.1f} min, {len(errors)} errors", flush=True)
        run.summary({"normalized": len(merged) - len(errors), "errors": len(errors)})
        run.table("prepare/errors", ["sample_id", "error"], [[e["sample_id"], e["error"]] for e in errors[:500]])
        ok = [m for m in merged.values() if "error" not in m]
        run.hist("prepare/short_side_px", [m["short_side"] for m in ok])
        run.hist("prepare/lead_silence_s", [m["lead_silence_s"] for m in ok])
        run.hist("prepare/duration_s", [m["duration_s"] for m in ok if m.get("duration_s")])
        run.finish()


def media_ok(cross, name):
    return {m["sample_id"]: m for m in read_jsonl(folder(cross, name) / "media.jsonl") if "error" not in m}


# ---------------------------------------------------------------- 3. extract (CPU stages, then GPU)
CPU_ENCODERS = ("landmarks", "mouth", "lipinc", "wave")


def step_extract(cross, only=None, session="all", devices=None):
    """session: "cpu" (landmarks, mouth, ...: CPU-only Kaggle session, no GPU quota), "gpu"
    (AV-HuBERT, XLS-R) or "all". Models of the roster decide which encoders are needed."""
    models = list(cross["models"].get("ours") or []) + list(cross["models"].get("published") or [])
    wanted = features.encoders_needed(models)
    if session == "cpu":
        wanted = [e for e in wanted if e in CPU_ENCODERS]
    elif session == "gpu":
        wanted = [e for e in wanted if e not in CPU_ENCODERS]
    for name in active_datasets(cross, only):
        cfg = main_cfg(cross, name)
        ok = media_ok(cross, name)
        rows = [r for r in read_jsonl(folder(cross, name) / "manifest.jsonl") if r["sample_id"] in ok]
        clips = {sid: m["clip"] for sid, m in ok.items()}
        run = open_run(cross, "extract", f"extract-{name}-{session}", group=name,
                       config={"dataset": name, "encoders": wanted, "samples": len(rows),
                               "encoder_specs": {e: cfg["encoders"].get(e) for e in wanted}})
        order = [e for e in ("landmarks", "avhubert", "mouth", "lipinc", "xlsr", "wave") if e in wanted]
        # mouth / lipinc need the landmark stage; on a GPU session they already ran on CPU.
        report = features.extract(cfg, rows, clips, order, devices=devices or [cfg.get("device")], run=run)
        write_json(folder(cross, name) / f"extract-{session}.json",
                   {e: {k: v for k, v in r.items()} for e, r in report.items()})
        run.summary({f"{e}/failed": len(r["failed"]) for e, r in report.items()}
                    | {f"{e}/elapsed_min": r["elapsed_s"] / 60 for e, r in report.items()})
        run.table("extract/failures", ["encoder", "sample_id", "error"],
                  [[e, s, m] for e, r in report.items() for s, m in list(r["failed"].items())[:300]])
        run.finish()


# ---------------------------------------------------------------- 4. score (GPU)
def conditions_for(model, cross):
    out = ["base"]
    if cross["protocols"].get("trim_leading_silence"):
        out.append("trim")
    if model is not None and getattr(model, "name", "") in ("trihead_v5",):
        allowed = set(cross["protocols"].get("knockout") or [])
        out += [f"knockout:{s}" for s in scoring.knockout_streams(model) if s in allowed]
    return out


def step_score(cross, only=None, device=None, workers=2):
    import torch
    from vn_av_df.training.data import ENCODERS_FOR_NEED

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    first = next(iter(active_datasets(cross, only)), None)
    if first is None:
        print("no dataset to score", flush=True)
        return
    roster = scoring.load_roster(cross, main_cfg(cross, first), device)
    write_json(Path(cross["work"]) / "results" / "roster.json",
               {k: {kk: vv for kk, vv in e.items() if kk != "ensemble"} | {"ensemble": e.get("ensemble")}
                for k, _, e in roster})
    for name in active_datasets(cross, only):
        cfg = main_cfg(cross, name, device)
        cfg["training"]["workers"] = workers
        ok = media_ok(cross, name)
        rows = [r for r in read_jsonl(folder(cross, name) / "manifest.jsonl") if r["sample_id"] in ok]
        lead = {sid: m.get("lead_silence_s", 0.0) for sid, m in ok.items()}
        out_dir = folder(cross, name) / "scores"
        out_dir.mkdir(parents=True, exist_ok=True)
        run = open_run(cross, "score", f"score-{name}", group=name,
                       config={"dataset": name, "device": device, "roster": [k for k, _, _ in roster],
                               "protocols": cross["protocols"]})
        scored = {}
        for key, model, entry in roster:
            if model is None:
                continue
            encs = sorted({"avhubert"} | {e for need in model.needs for e in ENCODERS_FOR_NEED.get(need, ())})
            have = features.available(cfg, rows, encs)
            subset = [r for r in rows if r["sample_id"] in have]
            conditions = conditions_for(model, cross)
            tick = time.time()

            def progress(n, key=key, total=len(subset)):
                if n % 100 == 0 or n == total:
                    run.log({f"score/{key}/done": n})

            values, errors = scoring.score_model(model, cfg, subset, device, conditions, lead,
                                                 cross["protocols"], workers=workers, progress=progress)
            scored[key] = values
            write_json(out_dir / f"{key}.json", {"key": key, "conditions": conditions, "scores": values,
                                                 "errors": errors, "missing_features": len(rows) - len(subset)})
            print(f"{name} / {key}: {len(values)} scored, {len(errors)} errors, "
                  f"{len(rows) - len(subset)} without features, {(time.time() - tick) / 60:.1f} min", flush=True)
            run.summary({f"{key}/scored": len(values), f"{key}/errors": len(errors),
                         f"{key}/missing_features": len(rows) - len(subset)})
        for key, model, entry in roster:
            if model is not None or "ensemble" not in entry:
                continue
            seed = key.rsplit("_seed", 1)[1]
            spec = entry["ensemble"]
            v, a = scored.get(f"{spec['visual']}_seed{seed}"), scored.get(f"{spec['audio']}_seed{seed}")
            if v is None or a is None:
                print(f"{key}: members missing, skipped", flush=True)
                continue
            values = scoring.ensemble_scores(spec, v, a)
            write_json(out_dir / f"{key}.json", {"key": key, "conditions": ["base", "trim"], "scores": values,
                                                 "errors": {}, "missing_features": 0})
        run.finish()
