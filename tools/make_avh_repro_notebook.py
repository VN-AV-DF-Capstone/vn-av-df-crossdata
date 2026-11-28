"""Generate notebooks/avh_align_repro.ipynb: AVH-Align (Smeu et al., CVPR 2025) re-scored the way the
authors' code does it, to explain why our zero-shot AVH-Align AUC (0.51-0.69) sits far below the paper.

Three scorings of the released checkpoint (AVH-Align_AV1M.pt, commit 55401de), same videos:
- ours      : the crossdata protocol (our AV-HuBERT cache, 7 s windows, video = highest window), read
              from scores/avh_align_official.json of the GPU sessions.
- rescored  : our cached AV-HuBERT features, scored with eval.py of the authors (L2 norm per frame,
              video = logsumexp(-output) over the whole video). Changes the scoring rule only.
- original  : the authors' pipeline end to end on a subset: deepfake_preprocess.preprocess_video (dlib on
              every frame, landmark interpolation, 96 px mouth ROI) on the original video,
              deepfake_feature_extraction (one AV-HuBERT pass over the whole video, audio-only and
              video-only), eval.process_video. Changes features and scoring.

python tools/make_avh_repro_notebook.py [--out notebooks/avh_align_repro.ipynb] [--kaggle]
--kaggle also saves it (not run) as linhxm/vn-av-df-crossdata-avh-align-repro with its inputs attached.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# AVH-Align: chấm lại theo đúng code của tác giả

Trong crossdata, AVH-Align bản tác giả (`AVH-Align_AV1M.pt`) chỉ đạt AUC 0,51-0,69, thấp hơn nhiều so với bài gốc (Smeu et al., CVPR 2025). Notebook này tách hai nguyên nhân có thể:

| Cách chấm | Đặc trưng AV-HuBERT | Điểm video | Số video |
|---|---|---|---|
| `ours` | cache crossdata: chia đoạn 8 s, bỏ frame mất mặt | max theo cửa sổ 7 s | toàn bộ mẫu |
| `rescored` | cùng cache | `eval.py` gốc: chuẩn hoá L2, logsumexp(-output) trên cả video | toàn bộ mẫu |
| `original` | `deepfake_preprocess.py` + `deepfake_feature_extraction.py` gốc trên video gốc: cả video một lượt, nội suy landmark | `eval.py` gốc | tập con `PER_CLASS` mỗi lớp |

- Code gốc lấy từ github.com/bit-ml/AVH-Align commit `55401de` (giấy phép CC BY-NC-SA 4.0), chạy trong môi trường AV-HuBERT của repo chính.
- Phiên CPU (Accelerator None) không tốn quota GPU; ước tính 3-5 giờ với mặc định. Có GPU thì tự dùng GPU.
- Input: Output của 3 notebook `vn-av-df-crossdata-*-gpu` (cache + điểm), 3 notebook `*-cpu` (manifest dự phòng) và 3 dataset gốc.
- **LAV-DF:** ở ô Input, chọn version 3 của `vn-av-df-crossdata-lavdf-gpu` (version 4 là bản lưu rỗng).
- W&B project `vn-av-df-crossdata`, group `avh-align-repro`; chỉ ghi số và biểu đồ.
"""

SETUP = """
import json, os, shutil, subprocess, sys, time
from pathlib import Path

MAIN_URL = "https://github.com/VN-AV-DF-Capstone/vn-av-df-forensics.git"
CROSS_URL = "https://github.com/VN-AV-DF-Capstone/vn-av-df-crossdata.git"
AVH_URL, AVH_COMMIT = "https://github.com/bit-ml/AVH-Align.git", "55401ded7d94a6ad59106c401696d89ce8d16faf"
MAIN, CROSS, AVH = Path("/kaggle/working/vn-av-df-forensics"), Path("/kaggle/working/vn-av-df-crossdata"), Path("/kaggle/temp/AVH-Align")
WORKERS, OUT = Path("/kaggle/temp/vn-av-df-workers"), Path("/kaggle/working/avh_repro")
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def logged(name, command, heartbeat=60, limit=None, **kw):
    # Run a command into its own log; print its last line every `heartbeat` s, the log tail on failure.
    path, start = LOGS / f"{name}.log", time.time()
    with open(path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, **kw)
        while True:
            try:
                code = proc.wait(timeout=heartbeat)
                break
            except subprocess.TimeoutExpired:
                tail = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:] or [""]
                print(f"[{name}] {time.time() - start:.0f} s: {tail[0][:200]}", flush=True)
                if limit and time.time() - start > limit:
                    proc.kill()
                    code = -9
                    break
    if code:
        print("\\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]))
        raise RuntimeError(f"{name} failed ({code}), log: {path}")
    print(f"[{name}] done in {time.time() - start:.0f} s", flush=True)


for url, path in ((MAIN_URL, MAIN), (CROSS_URL, CROSS)):
    if not path.exists():
        logged(f"clone-{path.name}", ["git", "clone", "-q", url, path])
if not AVH.exists():
    logged("clone-avh", ["git", "clone", "-q", AVH_URL, AVH])
    logged("checkout-avh", ["git", "-C", AVH, "checkout", "-q", AVH_COMMIT])
for name in ("weights", "external"):  # large assets outside the 20 GB Output
    (Path("/kaggle/temp/vn-av-df-heavy") / name).mkdir(parents=True, exist_ok=True)
    if not (MAIN / name).exists():
        (MAIN / name).symlink_to(Path("/kaggle/temp/vn-av-df-heavy") / name, target_is_directory=True)
sys.path[:0] = [str(MAIN), str(CROSS / "src")]
logged("pip", [sys.executable, "-m", "pip", "install", "-q", "-e", str(MAIN) + "[data]", "wandb"])
import settings
settings.bootstrap()
import torch
GPU = torch.cuda.is_available()
print("GPU:", GPU, "| CPUs:", os.cpu_count())
"""

WANDB = """
USE_WANDB = True   # secret WANDB_API_KEY (Add-ons -> Secrets)
if USE_WANDB:
    from kaggle_secrets import UserSecretsClient
    try:
        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as exc:
        raise RuntimeError("Bật secret WANDB_API_KEY cho notebook này, hoặc đặt USE_WANDB = False") from exc
    os.environ["WANDB_DIR"] = "/kaggle/temp"
"""

CONFIG = """
PER_CLASS = 200      # videos per class for the `original` pipeline (0 = skip it, rescoring only)
TRIM = False         # also run the authors' --trimmed features (leading silence removed): doubles that time
SEED = 2026
NB, DS = Path("/kaggle/input/notebooks/linhxm"), Path("/kaggle/input/datasets")
DATASETS = {  # key: (GPU-session Output, CPU-session Output = manifest fallback)
    "avdeepfake1m": ("vn-av-df-crossdata-avdeepfake1m-gpu", "vn-av-df-crossdata-avdeepfake1m-cpu"),
    "lavdf": ("vn-av-df-crossdata-lavdf-gpu", "vn-av-df-crossdata-lavdf-cpu"),
    "polyglotfake": ("vn-av-df-crossdata-polyglotfake-gpu", "vn-av-df-crossdata-polyglotfake-cpu"),
}
DISPLAY = {"avdeepfake1m": "AV-Deepfake1M (val)", "lavdf": "LAV-DF (test)", "polyglotfake": "PolyGlotFake"}
# Paper context (Smeu et al., CVPR 2025, Table 2, AUC %): AV-Deepfake1M 85.9 (trim 83.5), FakeAVCeleb 94.6.
PAPER = {"avdeepfake1m": 85.9}


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


sources = {}
for key, (gpu, cpu) in DATASETS.items():
    g, c = NB / gpu / "crossdata" / key, NB / cpu / "crossdata" / key
    manifest = g / "manifest.jsonl" if (g / "manifest.jsonl").is_file() else c / "manifest.jsonl"
    if not manifest.is_file():
        print(f"WARNING {key}: no manifest in {g} or {c}, skipped")
        continue
    cache = g / "cache" / "avhubert"
    scores = g / "scores" / "avh_align_official.json"
    sources[key] = {"manifest": manifest, "cache": cache if cache.is_dir() else None,
                    "scores": scores if scores.is_file() else None}
    print(key, "| manifest:", manifest.parent.parent.parent.name, "| cache:", bool(sources[key]["cache"]),
          "| ours:", bool(sources[key]["scores"]))
"""

ASSETS = """
# AV-HuBERT Large (self_large_vox_433h, the checkpoint AVH-Align uses) + dlib files + worker env (fairseq).
from vn_av_df.cli import setup
cfg0 = settings.config()
enc = cfg0["encoders"]["avhubert"]
if PER_CLASS:
    print(setup(cfg0, generators=[], encoders=["avhubert"], models=False))
    logged("worker-avhubert", [sys.executable, str(MAIN / "environments/bootstrap_worker.py"), "avhubert",
                               "--root", str(MAIN), "--env-root", str(WORKERS)] + ([] if GPU else ["--allow-cpu"]))
    WPY = str(WORKERS / ".venv-avhubert/bin/python")
    # Extra packages of the authors' scripts (librosa, scikit-video), numpy kept at the worker's pin.
    logged("pip-worker", [WPY, "-m", "pip", "install", "-q", "librosa==0.10.2.post1", "scikit-video==1.1.11",
                          "tqdm", "numpy==1.23.5"])
    if not Path("/usr/bin/ffmpeg").exists():  # the authors' preprocessing calls /usr/bin/ffmpeg
        import imageio_ffmpeg
        os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), "/usr/bin/ffmpeg")
    print("ffmpeg:", shutil.which("ffmpeg"), "| ffprobe (skvideo):", shutil.which("ffprobe"))
    print("AV-HuBERT:", enc["checkpoint"], "| repo:", enc["repo"])
"""

DRIVER = r'''
"""Authors' AVH-Align feature pipeline on a job list, inside the AV-HuBERT worker env.
python driver.py jobs.json: preprocess (deepfake_preprocess.preprocess_video) then features
(deepfake_feature_extraction: load_audio, load_transforms, extract_finetune over the whole video)."""
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

job = json.load(open(sys.argv[1], encoding="utf-8"))
AVHUBERT = Path(job["avhubert_dir"])
os.chdir(AVHUBERT)
sys.path[:0] = [str(AVHUBERT), str(AVHUBERT.parent / "fairseq"), job["avh_repo"]]
argv, sys.argv = sys.argv, sys.argv[:1]  # AV-HuBERT modules pick absolute imports when argv has one item
import numpy as np
import torch
import deepfake_preprocess as P
sys.argv = argv


def preprocess(item):
    src = Path(item["video"])
    out = Path(item["work"])
    roi = out / (src.name[:-4] + "_roi.mp4")
    if roi.is_file() and (out / (src.name[:-4] + ".wav")).is_file():
        return item["sid"], None
    try:
        ok = P.preprocess_video(str(src.parent), src.name, str(out), job["predictor"], job["mean_face"])
        return item["sid"], None if ok else "preprocess_video returned False"
    except Exception as exc:
        return item["sid"], f"{type(exc).__name__}: {exc}"


def main():
    items, errors = job["items"], {}
    tick = time.time()
    with ProcessPoolExecutor(int(job["workers"])) as pool:
        futures = [pool.submit(preprocess, it) for it in items]
        for n, f in enumerate(as_completed(futures), 1):
            sid, err = f.result()
            if err:
                errors[sid] = err
            if n % 25 == 0 or n == len(items):
                print(f"preprocess {n}/{len(items)}, {len(errors)} errors, {(time.time() - tick) / 60:.1f} min", flush=True)

    argv2, sys.argv = sys.argv, sys.argv[:1]
    import deepfake_feature_extraction as X
    sys.argv = argv2
    import utils as avhubert_utils
    from fairseq import checkpoint_utils

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 1)
    # = X.load_model without the hard-coded .cuda()
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([job["checkpoint"]])
    model = models[0]
    if hasattr(model, "decoder"):
        model = model.encoder.w2v_model
    model.to(device).eval()
    transform = X.load_transforms(task)
    tick = time.time()
    for n, it in enumerate(items, 1):
        for trimmed in [False, True] if job["trim"] else [False]:
            target = Path(it["features"] + ("_trim" if trimmed else "") + ".npz")
            if it["sid"] in errors or target.is_file():
                continue
            src = Path(it["video"])
            roi, wav = Path(it["work"]) / (src.name[:-4] + "_roi.mp4"), Path(it["work"]) / (src.name[:-4] + ".wav")
            try:
                # = X.extract_features (device-agnostic; the unused multimodal pass is skipped)
                frames = avhubert_utils.load_video(str(roi))
                frames = transform(frames)
                frames = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).to(device)
                silence = X.compute_starting_silence(str(wav)) if trimmed else 0
                audio = X.load_audio(str(wav), silence_duration=silence)[None, :, :].transpose(1, 2).to(device)
                skip = int(silence * X.FPS) + 1 if silence > 0 else 0
                frames = frames[:, :, skip:]
                length = min(frames.shape[2], audio.shape[-1])
                frames, audio = frames[:, :, :length], audio[:, :, :length]
                with torch.no_grad():
                    fa, _ = model.extract_finetune({"video": None, "audio": audio}, None, None)
                    fv, _ = model.extract_finetune({"video": frames, "audio": None}, None, None)
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, visual=fv.squeeze(0).cpu().numpy(), audio=fa.squeeze(0).cpu().numpy())
            except Exception as exc:
                errors[it["sid"] + ("_trim" if trimmed else "")] = f"{type(exc).__name__}: {exc}"[:300]
                traceback.print_exc()
        if n % 25 == 0 or n == len(items):
            spent = time.time() - tick
            print(f"features {n}/{len(items)} on {device}, {len(errors)} errors, {spent / 60:.1f} min, "
                  f"~{spent / n * (len(items) - n) / 60:.0f} min left", flush=True)
    json.dump(errors, open(job["errors"], "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
'''

RUN = """
import numpy as np
from vnav_cross.archive import extract as extract_member

sys.path.insert(0, str(AVH))
import importlib
avh_model, avh_eval = importlib.import_module("model"), importlib.import_module("eval")  # authors' model.py / eval.py
device = "cuda" if GPU else "cpu"
ckpt = torch.load(AVH / "checkpoints/AVH-Align_AV1M.pt", map_location=device, weights_only=False)
fusion = avh_model.FusionModel().to(device)
fusion.load_state_dict(ckpt["state_dict"])
fusion.eval()


def score(data):  # authors' eval.process_video: L2 norm per frame, logsumexp(-output) over the video
    with torch.no_grad():
        return float(avh_eval.process_video(data, fusion, device))


(OUT / "driver.py").write_text(DRIVER, encoding="utf-8")
rows_of, scores = {}, {}
for key, src in sources.items():
    rows = read_jsonl(src["manifest"])
    rows_of[key] = rows
    scores[key] = {"ours": {}, "rescored": {}, "original": {}, "original_trim": {}}
    if src["scores"]:
        ours = json.load(open(src["scores"], encoding="utf-8"))["scores"]
        scores[key]["ours"] = {sid: v["base"]["ai"] for sid, v in ours.items() if "base" in v}
    if src["cache"]:  # rescored: our cache, authors' scoring, valid frames only (invalid ones are zero vectors)
        tick, skipped = time.time(), 0
        for r in rows:
            f = src["cache"] / f"{r['sample_id']}.npz"
            if not f.is_file():
                skipped += 1
                continue
            with np.load(f) as z:
                valid = z["visual_valid"].astype(bool)
                if valid.sum() == 0:
                    skipped += 1
                    continue
                scores[key]["rescored"][r["sample_id"]] = score({"visual": z["visual"][valid].astype(np.float32),
                                                                 "audio": z["audio"][valid].astype(np.float32)})
        print(f"{key}: rescored {len(scores[key]['rescored'])} videos ({skipped} without features) in {time.time() - tick:.0f} s")

    if not PER_CLASS:
        continue
    rng = np.random.default_rng(SEED)
    by_class = {}
    for r in rows:
        by_class.setdefault(r["category"] if key != "polyglotfake" else ("fake" if r["label"] else "real"), []).append(r)
    subset = [r for cls in sorted(by_class) for r in
              (lambda g: [g[i] for i in sorted(rng.choice(len(g), min(PER_CLASS, len(g)), replace=False))])(by_class[cls])]
    work = Path("/kaggle/temp/avh_repro") / key
    items = []
    for r in subset:
        video = Path(r["path"])
        if r.get("archive"):  # AV-Deepfake1M val: pull the member out of the zip volumes
            video = work / "raw" / r["sample_id"] / Path(r["member"]).name
            if not video.is_file():
                extract_member(r["archive"], r["member"], video)
        items.append({"sid": r["sample_id"], "video": str(video), "work": str(work / "pre" / r["sample_id"]),
                      "features": str(work / "features" / r["sample_id"])})
    job = {"avhubert_dir": str(Path(enc["repo"]) / "avhubert"), "avh_repo": str(AVH), "checkpoint": enc["checkpoint"],
           "predictor": enc["landmarks"], "mean_face": enc["mean_face"], "trim": TRIM, "items": items,
           # skvideo.io.vread holds the whole video in RAM: 720p PolyGlotFake gets fewer processes
           "workers": 2 if key == "polyglotfake" else os.cpu_count(), "errors": str(work / "errors.json")}
    (work / "jobs.json").parent.mkdir(parents=True, exist_ok=True)
    (work / "jobs.json").write_text(json.dumps(job), encoding="utf-8")
    print(f"{key}: original pipeline on {len(items)} videos ({PER_CLASS} per class)", flush=True)
    logged(f"original-{key}", [WPY, OUT / "driver.py", work / "jobs.json"], heartbeat=120)
    for it in items:
        for name, suffix in (("original", ""), ("original_trim", "_trim")):
            f = Path(it["features"] + suffix + ".npz")
            if f.is_file():
                with np.load(f) as z:
                    scores[key][name][it["sid"]] = score({"visual": z["visual"], "audio": z["audio"]})
    errors = json.load(open(work / "errors.json", encoding="utf-8"))
    print(f"{key}: original scored {len(scores[key]['original'])}, errors {len(errors)}")
    json.dump(errors, open(OUT / f"errors_{key}.json", "w", encoding="utf-8"), indent=1)
json.dump(scores, open(OUT / "scores.json", "w", encoding="utf-8"))
"""

RESULTS = """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def auc_ci(rows, s, n_boot=1000, seed=SEED):
    # AUC on the videos that have a score, 95% CI by bootstrap over subjects (as in crossdata)
    rows = [r for r in rows if r["sample_id"] in s]
    y, x = np.array([r["label"] for r in rows]), np.array([s[r["sample_id"]] for r in rows])
    if len(set(y)) < 2:
        return None
    subj = np.array([r["subject"] for r in rows])
    groups = {g: np.flatnonzero(subj == g) for g in np.unique(subj)}
    keys, rng, boots = list(groups), np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.concatenate([groups[keys[i]] for i in rng.integers(len(keys), size=len(keys))])
        if len(set(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], x[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(rows), "n_fake": int(y.sum()), "auc": roc_auc_score(y, x), "lo": lo, "hi": hi}


table = []
for key, per in scores.items():
    rows = rows_of[key]
    sub = set(per["original"])  # the subset: every scoring is also measured on exactly these videos
    for name, s in per.items():
        if not s:
            continue
        scopes = [("subset", sub)] if name.startswith("original") else [("all", None)] + ([("subset", sub)] if sub else [])
        for scope, keep in scopes:
            m = auc_ci(rows, {k: v for k, v in s.items() if keep is None or k in keep})
            if m:
                table.append({"dataset": DISPLAY[key], "scoring": name, "videos": scope, **m,
                              "paper_auc": PAPER.get(key, np.nan) / 100})
    if key == "avdeepfake1m":  # the authors' own 10k AV-Deepfake1M test list (val split): videos in both
        tails = set(pd.read_csv(AVH / "av1m_metadata/test_metadata.csv")["path"])
        listed = {r["sample_id"] for r in rows if "/".join(str(r.get("member") or r["relative"]).split("/")[-4:]) in tails}
        print(f"AV-Deepfake1M: {len(listed)} of our videos are in the authors' 10k test list")
        for name, s in per.items():
            m = auc_ci(rows, {k: v for k, v in s.items() if k in listed}) if s and listed else None
            if m:
                table.append({"dataset": DISPLAY[key], "scoring": name, "videos": "authors' test list", **m,
                              "paper_auc": PAPER.get(key, np.nan) / 100})
res = pd.DataFrame(table)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
res.to_csv(OUT / "avh_repro_auc.csv", index=False)

# Duration effect: logsumexp over the whole video grows with the number of frames.
dur_rows = []
for key, per in scores.items():
    media = {}
    mfile = sources[key]["manifest"].with_name("media.jsonl")
    if mfile.is_file():
        media = {m["sample_id"]: m.get("duration_s") for m in read_jsonl(mfile) if "error" not in m}
    for name, s in per.items():
        common = [k for k in s if media.get(k)]
        if len(common) > 20:
            dur_rows.append({"dataset": DISPLAY[key], "scoring": name,
                             "spearman_score_duration": pd.Series([s[k] for k in common]).corr(
                                 pd.Series([media[k] for k in common]), method="spearman")})
dur = pd.DataFrame(dur_rows)
print(dur.round(3).to_string(index=False))

fig, axes = plt.subplots(1, len(scores), figsize=(5 * len(scores), 4.4), squeeze=False)
style = {"ours": ("#0072B2", "-"), "rescored": ("#E69F00", "-"), "original": ("#009E73", "-"), "original_trim": ("#009E73", "--")}
for ax, (key, per) in zip(axes[0], scores.items()):
    rows = {r["sample_id"]: r["label"] for r in rows_of[key]}
    sub = set(per["original"]) or None
    for name, s in per.items():
        ids = [k for k in s if (sub is None or k in sub) and k in rows]
        if len(set(rows[k] for k in ids)) < 2:
            continue
        fpr, tpr, _ = roc_curve([rows[k] for k in ids], [s[k] for k in ids])
        color, ls = style[name]
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f"{name} ({roc_auc_score([rows[k] for k in ids], [s[k] for k in ids]):.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls=":")
    ax.set_title(DISPLAY[key] + (" (subset)" if sub else ""))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
fig.suptitle("AVH-Align (authors' checkpoint): our protocol vs the authors' scoring")
fig.tight_layout()
fig.savefig(OUT / "avh_repro_roc.png", dpi=150)
plt.show()

if USE_WANDB:
    import wandb
    run = wandb.init(project="vn-av-df-crossdata", entity="vn-av-df-capstone", group="avh-align-repro",
                     job_type="avh-align-repro", name=time.strftime("%Y%m%d-%H%M") + " avh-align-repro",
                     config={"per_class": PER_CLASS, "trim": TRIM, "seed": SEED, "avh_commit": AVH_COMMIT,
                             "checkpoint": "AVH-Align_AV1M.pt", "device": device, "datasets": list(scores)})
    run.log({"auc": wandb.Table(dataframe=res.round(4)), "duration_correlation": wandb.Table(dataframe=dur.round(4)),
             "roc": wandb.Image(str(OUT / "avh_repro_roc.png"))})
    for r in table:
        run.summary[f"{r['dataset']}/{r['scoring']}/{r['videos']}/auc"] = r["auc"]
    art = wandb.Artifact("avh-align-repro", type="results")
    for f in ("avh_repro_auc.csv", "avh_repro_roc.png"):
        art.add_file(str(OUT / f))
    run.log_artifact(art)
    run.finish()
for d in Path("/kaggle/temp/avh_repro").glob("*/raw"):
    shutil.rmtree(d, ignore_errors=True)
print("outputs:", sorted(p.name for p in OUT.iterdir()))
"""


def cell(kind, text, cid):
    text = text.strip("\n")
    return {"cell_type": kind, "id": cid, "metadata": {}, "source": text.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {})}


def build():
    driver = "DRIVER = " + repr(DRIVER)
    cells = [cell("markdown", INTRO, "intro"), cell("code", SETUP, "setup"), cell("code", WANDB, "wandb"),
             cell("code", CONFIG, "config"), cell("code", ASSETS, "assets"), cell("code", driver, "driver"),
             cell("code", RUN, "run"), cell("code", RESULTS, "results")]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "notebooks/avh_align_repro.ipynb"))
    parser.add_argument("--kaggle", action="store_true", help="save the notebook on Kaggle (quick save, not run)")
    parser.add_argument("--owner", default="linhxm")
    args = parser.parse_args()
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print("wrote", args.out)
    if args.kaggle:
        import sys

        from kaggle.api.kaggle_api_extended import KaggleApi

        sys.path.insert(0, str(ROOT / "tools"))
        from create_kaggle_notebooks import DATASETS, save

        names = ("avdeepfake1m", "lavdf", "polyglotfake")
        spec = {"slug": "vn-av-df-crossdata-avh-align-repro", "gpu": False,
                "datasets": [DATASETS[n] for n in names],
                "kernels": [f"{args.owner}/vn-av-df-crossdata-{n}-{kind}" for kind in ("cpu", "gpu") for n in names]}
        api = KaggleApi()
        api.authenticate()
        url, error = save(api, args.owner, spec, text)
        print("ERROR" if error else "saved", spec["slug"], url or "", error or "")


if __name__ == "__main__":
    main()
