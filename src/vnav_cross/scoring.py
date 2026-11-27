"""Load the detectors and score cached features: windowed, with the test-time protocol variants.

Roster
- ours: checkpoints trained on VN-AV-DF (`<run>/<model>_seed<seed>/best.pt`), thresholds taken
  from their `metrics.json` (5 % FPR on VN-AV-DF validation reals, like the demo); BL-Ens-AV is
  rebuilt from its two members and `ensemble.json` (z-score stats of VN-AV-DF validation reals).
- published: `avh_align_official` = AVH-Align with the authors' released checkpoint (45k
  AV-Deepfake1M real videos), no VN-AV-DF data at all; it has no VN threshold.

Scoring (same as vn_av_df.demo.detect): 7 s windows with half overlap, video score per task = the
highest window score. Raw scores are stored; the resolution gate is applied at evaluation time,
so gated and ungated results come from one forward pass.

Protocol variants, all from one feature cache:
- "base": the normalized video as is;
- "trim": the leading silence (media.leading_silence) cut from every frame stream, i.e. the
  AVH-Align shortcut check. Features are sliced, not re-extracted: AV-HuBERT context near the cut
  differs slightly from a re-encoded trimmed file (stated in the report);
- "knockout:<stream>" (M3-R only): one branch output set to zero before the heads. This is a
  test-time ablation of the trained model (what the heads rely on), not a retrained ablation.
"""

import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

DISPLAY = {
    "trihead_v5": "M3-R",
    "trihead_v5_no_pair": "M3-R w/o pairing",
    "trihead_v5_clean": "M3-R clean-only",
    "trihead_v5_quality": "M3-R + quality",
    "avh_align_plus": "M2",
    "avh_align_sup": "BL-M/sup",
    "avh_align": "BL-M",
    "xlsr_frozen": "BL-A2",
    "speechforensics": "SpeechForensics",
    "lipinc_v2": "BL-V",
    "bl_ens_av": "BL-Ens-AV",
    "avh_align_official": "AVH-Align (official)",
}
ROLE = {
    "trihead_v5": "ours: lip-audio-head consistency, three heads (visual / audio / mismatch)",
    "avh_align_plus": "ours: AVH-Align scorer over a +-15 lag volume + lip-head motion",
    "avh_align_sup": "baseline: AVH-Align/sup re-trained on VN-AV-DF",
    "avh_align": "baseline: AVH-Align (unsupervised) fine-tuned on VN-AV-DF reals",
    "xlsr_frozen": "baseline: frozen XLS-R 300M audio expert",
    "speechforensics": "published method, training-free (AV-HuBERT cosine)",
    "lipinc_v2": "baseline: LIPINC-V2 mouth-region transformer",
    "bl_ens_av": "baseline: late fusion max(z(BL-M/sup), z(BL-A2))",
    "avh_align_official": "published weights: AVH-Align checkpoint trained on AV-Deepfake1M reals",
}
KNOCKOUT_STREAMS = ("trace", "sync_rel", "sync_abs", "audio", "phi")


def display(key):
    name, _, seed = key.partition("_seed")
    return DISPLAY.get(name, name) + (f" (s{seed})" if seed else "")


# ---------------------------------------------------------------- roster
def thresholds_of(metrics):
    return {
        task: (m.get("operating_points", {}).get("fpr_0.05") or {}).get("threshold", m.get("threshold"))
        for task, m in metrics.get("tasks", {}).items()
    }


def in_domain(metrics):
    """VN-AV-DF test AUC (clean and re-encoded) from the checkpoint's metrics.json."""
    out = {"clean": (metrics.get("tasks", {}).get("ai") or {}).get("auc")}
    for name, entry in (metrics.get("variants") or {}).items():
        auc = ((entry.get("tasks") or {}).get("ai") or {}).get("auc")
        if auc is not None:
            out[name] = auc
    return out


def discover(runs, wanted):
    """{key: folder} for `<model>_seed<seed>` folders under every run root (first root wins)."""
    found = {}
    for root in [Path(r) for r in runs]:
        if not root.is_dir():
            print(f"run root not found, skipped: {root}", flush=True)
            continue
        for folder in sorted(root.iterdir()):
            name = folder.name.rsplit("_seed", 1)[0]
            if name not in wanted or folder.name in found:
                continue
            if (folder / "best.pt").is_file() or (folder / "ensemble.json").is_file():
                found[folder.name] = folder
    return found


def module_table(model):
    rows = []
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        if count:
            rows.append({"module": name, "type": type(module).__name__, "parameters": count})
    return rows


def describe(key, model=None, folder=None):
    """Architecture summary for the report and W&B (never the raw state dict)."""
    name = key.rsplit("_seed", 1)[0]
    info = {"key": key, "model": name, "display": display(key), "role": ROLE.get(name, "")}
    stored = folder and (Path(folder) / "architecture.json")
    if stored and stored.is_file():
        arch = json.loads(stored.read_text(encoding="utf-8"))
        info.update({k: arch.get(k) for k in ("class", "spec", "needs", "heads", "pool", "parameters", "modules")})
        info["training"] = {k: (arch.get("training") or {}).get(k) for k in ("epochs", "lr", "batch_size", "hidden")}
    elif model is not None:
        info.update({
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "needs": sorted(model.needs),
            "heads": dict(model.heads),
            "pool": model.pool,
            "parameters": sum(p.numel() for p in model.parameters()),
            "modules": module_table(model),
        })
    info["train_data"] = (
        "AV-Deepfake1M train reals (authors' release)" if name == "avh_align_official"
        else "none (training-free)" if name == "speechforensics"
        else "VN-AV-DF parts 1-3 train split"
    )
    return info


def load_roster(cross, main_cfg, device):
    """[(key, model or None, entry)] for every detector we can load; missing ones are reported."""
    from vn_av_df.training.models import build
    from vn_av_df.training.train import load_model

    ours = list(cross["models"].get("ours") or [])
    members = {"avh_align_sup", "xlsr_frozen"} if "bl_ens_av" in ours else set()
    found = discover(cross["models"].get("runs") or [], set(ours) | members)
    roster = []
    for key, folder in found.items():
        name = key.rsplit("_seed", 1)[0]
        metrics_path = folder / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        entry = {"folder": str(folder), "group": "ours", "thresholds": thresholds_of(metrics),
                 "in_domain": in_domain(metrics), "member_only": name not in ours}
        if name == "bl_ens_av":
            entry["ensemble"] = json.loads((folder / "ensemble.json").read_text(encoding="utf-8"))
            entry["describe"] = describe(key, None, None) | {"members": entry["ensemble"]}
            roster.append((key, None, entry))
            continue
        model = load_model(folder / "best.pt", device)
        entry["describe"] = describe(key, model, folder)
        roster.append((key, model, entry))
    if "avh_align_official" in (cross["models"].get("published") or []):
        path = main_cfg["model_assets"]["avh_align_init"]
        if Path(path).is_file():
            model = build("avh_align", dim=1024, assets={"avh_align_init": path, "load_pretrained": True})
            model = model.to(device).eval()
            roster.append(("avh_align_official", model, {
                "group": "published", "thresholds": {}, "in_domain": {}, "member_only": False,
                "describe": describe("avh_align_official", model)}))
        else:
            print(f"AVH-Align checkpoint missing ({path}); run vn-av-df-forensics training/00_setup.py", flush=True)
    missing = sorted(set(ours) - {k.rsplit("_seed", 1)[0] for k, _, _ in roster})
    if missing:
        print(f"not found in the run roots, skipped: {missing}", flush=True)
    return roster


# ---------------------------------------------------------------- test-time variants
@contextmanager
def knockout(model, stream):
    """Zero one branch output of a TriHead model (or the AVH-Align phi channel) during scoring."""
    if stream == "phi":
        phi = model.sync.phi
        original = phi.lag_volume
        phi.lag_volume = lambda v, a: torch.zeros(*v.shape[:2], 31, device=v.device, dtype=v.dtype)
        try:
            yield
        finally:
            phi.lag_volume = original
        return
    original = model.branches

    def patched(batch):
        feats = original(batch)
        feats[stream] = torch.zeros_like(feats[stream])
        return feats

    model.branches = patched
    try:
        yield
    finally:
        model.branches = original


def knockout_streams(model):
    if not hasattr(model, "branches") or not hasattr(model, "sync"):
        return []
    out = [s for s in KNOCKOUT_STREAMS if s != "phi"]
    if getattr(model.sync, "phi", None) is not None:
        out.append("phi")
    return out


def trim_item(item, frames):
    """Drop the first `frames` frames of every per-frame stream (static keys are kept)."""
    from vn_av_df.training.data import STATIC_KEYS

    if frames <= 0:
        return item
    length = len(item["valid"])
    if frames >= length - 25:  # keep at least 1 s
        frames = max(0, length - 25)
    return {k: v if k in STATIC_KEYS else v[frames:] for k, v in item.items()}


# ---------------------------------------------------------------- windowed scoring
def windows(length, window, stride):
    if length <= window:
        return [(0, length)]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] + window < length:
        starts.append(length - window)
    return [(s, s + window) for s in starts]


@torch.no_grad()
def score_item(model, item, device, window=175, stride=88):
    """{task: video score} = max over windows (one batch per video)."""
    from vn_av_df.demo.detect import blank_targets
    from vn_av_df.training.data import STATIC_KEYS, collate, to_device
    from vn_av_df.training.evaluate import task_scores
    from vn_av_df.training.models import pool_scores

    length = len(item["valid"])
    parts = []
    for i, (a, b) in enumerate(windows(length, window, stride)):
        part = {k: v if k in STATIC_KEYS else v[a:b] for k, v in item.items()}
        part = blank_targets(part, b - a)
        part["index"] = i
        parts.append(part)
    batch = to_device(collate(parts), device)
    heads = model(batch)
    mask = batch["mask"]
    best = {}
    for i in range(len(parts)):
        pooled = {
            h: float(pool_scores(v[i : i + 1], mask[i : i + 1], model.pool)[0])
            for h, v in heads.items()
            if not h.startswith("_")
        }
        for task, value in task_scores(model, {"video": pooled}).items():
            if value is not None and math.isfinite(value):
                best[task] = max(best.get(task, -np.inf), value)
    return best


class Loader(torch.utils.data.Dataset):
    """Parallel npz loading (BenchData.load) for rows that have every needed cache file."""

    def __init__(self, cfg, rows, needs):
        from vn_av_df.training.data import BenchData

        self.rows = rows
        self.data = BenchData(cfg, rows, needs)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            item, _ = self.data.load(self.rows[i])
            return self.rows[i]["sample_id"], item, None
        except Exception as exc:
            return self.rows[i]["sample_id"], None, f"{type(exc).__name__}: {str(exc)[:200]}"


def as_is(sample):
    return sample


def score_model(model, cfg, rows, device, conditions, lead, protocol, workers=2, progress=None):
    """{sample_id: {condition: {task: score}}} + {sample_id: error}. `lead`: {sample_id: seconds}."""
    loader = torch.utils.data.DataLoader(
        Loader(cfg, rows, model.needs), batch_size=None, shuffle=False, num_workers=workers,
        collate_fn=as_is,
    )
    window, stride = int(protocol.get("window", 175)), int(protocol.get("stride", 88))
    out, errors = {}, {}
    model.eval()
    for n, (sid, item, error) in enumerate(loader, 1):
        if item is None:
            errors[sid] = error
            continue
        if not item["valid"].any():
            errors[sid] = "no valid frame (face not tracked in any frame)"
            continue
        result = {}
        for condition in conditions:
            current = item
            if condition == "trim":
                current = trim_item(item, int(round(lead.get(sid, 0.0) * 25)))
            if condition.startswith("knockout:"):
                with knockout(model, condition.split(":", 1)[1]):
                    result[condition] = score_item(model, current, device, window, stride)
            else:
                result[condition] = score_item(model, current, device, window, stride)
        out[sid] = result
        if progress:
            progress(n)
    return out, errors


def ensemble_scores(spec, visual, audio):
    """BL-Ens-AV: ai = max(z_V, z_A) with VN-AV-DF validation-real stats (evaluate.zscore)."""
    (mv, sv), (ma, sa) = spec["stats"]["visual"], spec["stats"]["audio"]
    out = {}
    for sid in set(visual) & set(audio):
        out[sid] = {}
        for condition in set(visual[sid]) & set(audio[sid]):
            v, a = visual[sid][condition].get("visual"), audio[sid][condition].get("audio")
            if v is None or a is None:
                continue
            zv, za = (v - mv) / (sv + 1e-6), (a - ma) / (sa + 1e-6)
            out[sid][condition] = {"visual": zv, "audio": za, "ai": max(zv, za)}
    return out
