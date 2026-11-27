"""Pack the VN-AV-DF checkpoints of several seeds into one folder to upload as a private Kaggle dataset.

    python tools/export_weights.py <run seed 42> <run seed 43> --out ../kaggle_upload/vn-av-df-weights-2seeds

Each run folder is a `runs/train_all_v00x` of vn-av-df-forensics (Kaggle Output of the test session).
Per model and seed: best.pt + metrics.json (operating-point thresholds, VN-AV-DF test AUC) +
architecture.json. BL-Ens-AV has no checkpoint: ensemble.json stores its two members and the
mean / std of each member's scores on validation reals (vn_av_df.training.evaluate.zscore), and the
script checks that the 5 % FPR threshold rebuilt from those stats equals the one in metrics.json.
Downloadable weights (AV-HuBERT, XLS-R, the released AVH-Align checkpoint) are not packed.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

MODELS = ["trihead_v5", "trihead_v5_no_pair", "avh_align_plus", "avh_align_sup", "xlsr_frozen", "avh_align",
          "speechforensics", "lipinc_v2"]
ENSEMBLE = {"name": "bl_ens_av", "visual": "avh_align_sup", "audio": "xlsr_frozen"}
FILES = ("best.pt", "metrics.json", "architecture.json")


def load(path, task):
    with np.load(path, allow_pickle=False) as z:
        return dict(zip(z["sample_id"].tolist(), z[f"video_{task}"].tolist(), strict=True))


def seeds_in(run):
    return sorted({p.name.rsplit("_seed", 1)[1] for p in run.iterdir() if "_seed" in p.name and p.is_dir()})


def ensemble(run, out, seed):
    stats, val = {}, {}
    for task in ("visual", "audio"):
        scores = load(run / f"{ENSEMBLE[task]}_seed{seed}" / "predictions-val.npz", task)
        reals = np.array([v for k, v in scores.items() if k.endswith("_real")])
        mean, std = float(reals.mean()), float(reals.std() + 1e-6)
        stats[task] = [mean, std]
        val[task] = {k: (v - mean) / std for k, v in scores.items()}
    src = run / f"{ENSEMBLE['name']}_seed{seed}" / "metrics.json"
    # Check: 5 % FPR threshold of the fused score on validation videos without AI (real + sham).
    ids = sorted(set(val["visual"]) & set(val["audio"]))
    negatives = [max(val["visual"][k], val["audio"][k]) for k in ids if k.endswith("_real") or "_sham" in k]
    rebuilt = float(np.quantile(negatives, 0.95))
    saved = json.loads(src.read_text(encoding="utf-8"))["tasks"]["ai"]["operating_points"]["fpr_0.05"]["threshold"]
    print(f"bl_ens_av_seed{seed}: threshold rebuilt {rebuilt:.4f}, metrics.json {saved:.4f}")
    if abs(rebuilt - saved) > 1e-3:
        raise SystemExit(f"seed {seed}: z-score stats do not reproduce metrics.json, nothing exported")
    dst = out / f"{ENSEMBLE['name']}_seed{seed}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / "metrics.json")
    (dst / "ensemble.json").write_text(
        json.dumps({"visual": ENSEMBLE["visual"], "audio": ENSEMBLE["audio"], "stats": stats}, indent=1),
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="runs/train_all_v00x folders (one or more seeds each)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--owner", default="linhxm")
    parser.add_argument("--slug", default="vn-av-df-weights-2seeds")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    packed = []
    for run in map(Path, args.runs):
        for seed in seeds_in(run):
            for name in MODELS:
                src = run / f"{name}_seed{seed}"
                if not (src / "best.pt").is_file():
                    continue
                dst = out / src.name
                dst.mkdir(exist_ok=True)
                for file in FILES:
                    if (src / file).is_file():
                        shutil.copy2(src / file, dst / file)
                packed.append(src.name)
            if (run / f"{ENSEMBLE['name']}_seed{seed}" / "metrics.json").is_file():
                ensemble(run, out, seed)
                packed.append(f"{ENSEMBLE['name']}_seed{seed}")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "VN-AV-DF weights (2 seeds)", "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"{out}: {len(packed)} folders, {size:.1f} MB")
    print("\n".join(sorted(packed)))


if __name__ == "__main__":
    main()
