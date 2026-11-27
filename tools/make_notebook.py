"""Generate notebooks/crossdata.ipynb (Kaggle).

    python tools/make_notebook.py
    python tools/make_notebook.py --out smoke/crossdata.ipynb --set SESSION='"all"' --set ONLY='["vnavdf"]'

--set NAME=VALUE replaces the line `NAME = ...` of the setup / config cells (a smoke run or a
session-specific copy without editing the notebook by hand on Kaggle).
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text):
    source = text.strip("\n").splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


INTRO = """
# VN-AV-DF · Cross-dataset evaluation (M3-R, M2, baselines, published detectors)

Đo các detector train trên VN-AV-DF (tiếng Việt) **zero-shot** trên dataset công khai (FakeAVCeleb, AV-Deepfake1M, LAV-DF, MAVOS-DD), cạnh detector đã công bố chạy bằng **weight của tác giả** (AVH-Align checkpoint AV-1M; SpeechForensics không cần train). Kèm: chuyển ngưỡng VN → dataset mới, ngưỡng hiệu chỉnh lại trên 30 % người của dataset đích, so cặp với M3-R (bootstrap theo người), kiểm lối tắt khoảng lặng đầu (AVH-Align), cổng độ phân giải bật / tắt, ablation tắt nhánh của M3-R lúc test, và **audit rò rỉ** của VN-AV-DF. Kết quả hiển thị bằng tiếng Anh, log W&B project `vn-av-df-crossdata` (private).

| SESSION | Accelerator | Làm gì | Cần attach |
|---|---|---|---|
| `cpu` | None | lấy mẫu (manifest) → chuẩn hoá video (25 fps, ≤ 720p) + đo khoảng lặng đầu → landmark + crop miệng (CPU, không tốn quota GPU) | dataset ngoài (đã xin quyền, upload private) |
| `gpu` | T4 ×2 | AV-HuBERT + XLS-R → chấm mọi detector → evaluate → report | Output phiên `cpu` (`PREVIOUS_WORK`) + weight `linhxm/vn-av-df-demo-weights` (+ run seed 43 nếu có) |
| `report` | None | dựng lại bảng / biểu đồ / report.md từ điểm đã có | Output phiên `gpu` |
| `audit` | None | audit rò rỉ VN-AV-DF: trùng người / video nguồn / clip / hash / audio donor / câu giữa các split, số lần dùng test, lối tắt đặc trưng đơn | 3 dataset part VN-AV-DF |

Dataset ngoài đều phải xin quyền (form / HF gated, không phân phối lại): upload thành **Kaggle dataset private**, rồi sửa đường dẫn ở `DATASETS`. Dataset nào thiếu thì bỏ qua. Hết giờ giữa chừng: chạy lại cùng `SESSION`, attach Output phiên dở vào `PREVIOUS_WORK` (mẫu đã xong được bỏ qua).
"""

SETUP = """
from pathlib import Path
import subprocess, sys, os, time, json, shutil
SESSION_START = time.time()
MAIN_URL = "https://github.com/linhxm/vn-av-df-forensics.git"
MAIN = Path("/kaggle/working/vn-av-df-forensics")
# Repo này (public trên GitHub); clone lỗi thì dùng bản upload thành Kaggle dataset (nếu có attach).
CROSS_URL = "https://github.com/VN-AV-DF-Capstone/vn-av-df-crossdata.git"
CROSS_DATASET = Path("/kaggle/input/datasets/linhxm/vn-av-df-crossdata-code")
CROSS = Path("/kaggle/working/vn-av-df-crossdata")
WORKERS = Path("/kaggle/temp/vn-av-df-workers")
LOGS = Path("/kaggle/temp/vn-av-df-logs"); LOGS.mkdir(parents=True, exist_ok=True)
def logged(name, command, **kw):
    path = LOGS / f"{name}.log"
    with open(path, "a", encoding="utf-8") as log:
        code = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, **kw).returncode
    if code:
        print("\\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]))
        raise RuntimeError(f"{name} failed ({code}); log: {path}")
if not MAIN.exists():
    logged("git-main", ["git", "clone", MAIN_URL, str(MAIN)])
if not CROSS.exists():
    try:
        logged("git-cross", ["git", "clone", CROSS_URL, str(CROSS)])
    except RuntimeError:
        if not CROSS_DATASET.is_dir():
            raise
        print("clone failed, using the attached code dataset", CROSS_DATASET)
        shutil.copytree(CROSS_DATASET, CROSS, ignore=shutil.ignore_patterns("work", "__pycache__"))
for name in ("weights", "external"):  # weight lớn ở /kaggle/temp, không vào Output
    (Path("/kaggle/temp/vn-av-df-heavy") / name).mkdir(parents=True, exist_ok=True)
    if not (MAIN / name).exists():
        (MAIN / name).symlink_to(Path("/kaggle/temp/vn-av-df-heavy") / name, target_is_directory=True)
sys.path[:0] = [str(MAIN), str(CROSS / "src")]
logged("pip", [sys.executable, "-m", "pip", "install", "-e", str(MAIN) + "[data,training]", "-e", str(CROSS) + "[mavos]"])
import settings
settings.bootstrap()
os.environ["VNAVDF_ROOT"] = str(MAIN)
"""

WANDB = """
USE_WANDB = True   # secret WANDB_API_KEY (Add-ons → Secrets); project private, không đẩy ảnh / video người thật
if USE_WANDB:
    from kaggle_secrets import UserSecretsClient
    try:
        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as exc:
        raise RuntimeError("Bật secret WANDB_API_KEY cho notebook này (Add-ons → Secrets), "
                           "hoặc đặt USE_WANDB = False") from exc
    os.environ["WANDB_DIR"] = "/kaggle/temp"
"""

CONFIG = """
SESSION = "cpu"    # "cpu" → "gpu" → ("report"); "audit" độc lập; "all" = cpu + gpu trong một phiên (chạy thử nhỏ)
# Dataset ngoài: bản public trên Kaggle (Add Input → tìm đúng tên). Thiếu bộ nào thì bộ đó bị bỏ qua.
DATASETS = {
    "fakeavceleb":  "/kaggle/input/datasets/aicontentdetections/fakeavceleb-v1-2/FakeAVCeleb_v1.2",
    "avdeepfake1m": "/kaggle/input/datasets/aicontentdetections/av-deepfake1m/av-deepfake1m",
    "lavdf":        "/kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF",
    "polyglotfake": "/kaggle/input/datasets/debajyatidey/polyglotfake/PolyGlotFake",
}
ONLY = None        # vd. ["fakeavceleb"]: một dataset mỗi phiên nếu Output 20 GB không đủ
PER_STRATUM = None # ghi đè số video mỗi lớp (mặc định trong configs/crossdata.yaml: 500; MAVOS 60 mỗi ô)
# Checkpoint của mình: thư mục chứa <model>_seed<seed>/best.pt + metrics.json.
RUNS = ["/kaggle/input/datasets/linhxm/vn-av-df-weights-2seeds"]  # seed 42 + 43 (tools/export_weights.py)
# Output phiên trước (danh sách): phiên "gpu" = Output phiên "cpu" của cùng dataset; phiên "report" = Output
# mọi phiên "gpu" (chỉ chép manifest / media / scores, không chép cache) để gộp mọi dataset vào một báo cáo.
PREVIOUS_WORK = []    # vd. [Path("/kaggle/input/notebooks/linhxm/vn-av-df-crossdata-fakeavceleb-cpu/crossdata")]
WORK = Path("/kaggle/working/crossdata")
# audit: manifest 3 part VN-AV-DF + test-log của các vòng (trong repo chính)
VN_PARTS = [f"/kaggle/input/datasets/linhxm/{s}" for s in (
    "vn-av-df-data-part1-trim04-fake", "vn-av-df-data-part2-fake", "vn-av-df-data-part3-fake")]

import torch
from vnav_cross import pipeline
settings.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # settings.config() reads it at call time
cross = pipeline.load_config(work=str(WORK), main_root=str(MAIN))
for name, root in DATASETS.items():
    cross["datasets"][name]["root"] = root
    if PER_STRATUM:
        cross["datasets"][name]["sample"]["per_stratum"] = PER_STRATUM
cross["models"]["runs"] = RUNS
cross["wandb"] = {"mode": "online"} if USE_WANDB else None
LIGHT = shutil.ignore_patterns("cache", "clips", "raw", "figures", "tables")
for previous in map(Path, PREVIOUS_WORK if isinstance(PREVIOUS_WORK, (list, tuple)) else [PREVIOUS_WORK] if PREVIOUS_WORK else []):
    if not previous.is_dir():  # a notebook not run yet (or its Output not attached): skip that dataset
        print("WARNING: missing, skipped:", previous)
        continue
    # dirs_exist_ok: several sessions (one per dataset) merge into one work folder
    shutil.copytree(previous, WORK, dirs_exist_ok=True, ignore=LIGHT if SESSION == "report" else None)
    print("merged", previous)
GPU = torch.cuda.is_available()
if SESSION == "gpu" and not GPU:
    raise RuntimeError('SESSION = "gpu" cần Accelerator T4 ×2')
if SESSION in ("cpu", "report", "audit") and GPU:
    print(f"WARNING: SESSION {SESSION} only needs CPU; choose Accelerator None to save GPU quota")
print("datasets found:", list(pipeline.active_datasets(cross, ONLY)))
"""

ASSETS = """
# Tải đúng thứ phiên cần: CPU = landmark dlib; GPU = AV-HuBERT Large + XLS-R + checkpoint AVH-Align công bố.
from vn_av_df.cli import setup
from vnav_cross.features import encoders_needed
cfg0 = settings.config()
NEEDED = encoders_needed(cross["models"]["ours"] + cross["models"]["published"])
if SESSION in ("cpu", "gpu", "all"):
    ENCODERS = NEEDED if SESSION == "all" else [e for e in NEEDED if (e in pipeline.CPU_ENCODERS) == (SESSION == "cpu")]
    print("encoders this session:", ENCODERS)
    print(setup(cfg0, generators=[], encoders=ENCODERS, models=SESSION != "cpu"))
    if {"landmarks", "avhubert"} & set(ENCODERS):
        logged("worker-avhubert", [sys.executable, str(MAIN / "environments/bootstrap_worker.py"), "avhubert",
                                   "--root", str(MAIN), "--env-root", str(WORKERS)] + ([] if GPU else ["--allow-cpu"]))
        settings.AVH_PYTHON = str(WORKERS / ".venv-avhubert/bin/python")
"""

RUN = """
from vnav_cross.evaluate import step_evaluate
from vnav_cross.report import step_report
tick = time.time()
if SESSION == "cpu":
    pipeline.step_manifest(cross, ONLY)
    pipeline.step_prepare(cross, ONLY, workers=os.cpu_count())
    pipeline.step_extract(cross, ONLY, session="cpu")
elif SESSION in ("gpu", "all"):
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    if SESSION == "all":
        pipeline.step_manifest(cross, ONLY)
        pipeline.step_prepare(cross, ONLY, workers=os.cpu_count())
    pipeline.step_extract(cross, ONLY, session=SESSION, devices=devices)
    pipeline.step_score(cross, ONLY, device="cuda" if GPU else "cpu", workers=4)
    step_evaluate(cross, ONLY)
    step_report(cross)
elif SESSION == "report":
    step_evaluate(cross, ONLY)
    step_report(cross)
elif SESSION == "audit":
    from vnav_cross.audit import run_audit
    from vnav_cross.report import step_audit_report
    manifests = sorted(str(p) for root in VN_PARTS for p in Path(root).rglob("manifest.jsonl"))
    logs = [str(MAIN / f"docs/{r}_results/test-log.json") for r in ("v004", "v005", "v006")]
    audit = run_audit(manifests, logs, [str(Path(m).with_name("shortcut_audit.json")) for m in manifests])
    pipeline.write_json(WORK / "results/audit.json", audit)
    print(json.dumps(audit["verdict"], indent=1))
    step_audit_report(cross, audit, WORK / "results")
print(f"session {SESSION} done in {(time.time() - tick) / 3600:.2f} h")
"""

SHOW = """
# Hiển thị kết quả ngay trong notebook (tiếng Anh): kiến trúc, thông số, bảng, biểu đồ.
from IPython.display import Markdown, Image, display
results = WORK / "results"
for name in ("report.md", "audit.md"):
    if (results / name).is_file():
        text = (results / name).read_text(encoding="utf-8")
        display(Markdown(text.replace("](figures/", f"]({results}/figures/")))
for png in sorted((results / "figures").glob("*.png")):
    print(png.name)
    display(Image(filename=str(png)))
"""

CLEAN = """
# Output (20 GB) giữ: work/<dataset>/{manifest, media, cache, scores} và results; bỏ clip chuẩn hoá
# sau phiên GPU (đặc trưng đã trích, không cần lại) để vừa giới hạn.
if SESSION in ("gpu", "all"):
    for clips in WORK.glob("*/clips"):
        shutil.rmtree(clips, ignore_errors=True)
for path in ("vn-av-df-forensics", "vn-av-df-crossdata"):
    shutil.rmtree(Path("/kaggle/working") / path, ignore_errors=True)
print(subprocess.run(["du", "-sh", str(WORK)], capture_output=True, text=True).stdout)
"""


def override(text, sets):
    for name, value in sets.items():
        pattern = rf"(?m)^{name} = [^#\n]*?(\s*#.*)?$"
        text, count = re.subn(pattern, lambda m, n=name, v=value: f"{n} = {v}" + (m.group(1) or ""), text)
        if count:
            SEEN.add(name)
    return text


SEEN = set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "notebooks/crossdata.ipynb"))
    parser.add_argument("--set", action="append", default=[], help="NAME=VALUE (Python literal)")
    args = parser.parse_args()
    sets = dict(s.split("=", 1) for s in args.set)
    global SETUP, WANDB, CONFIG
    SETUP, WANDB, CONFIG = (override(x, sets) for x in (SETUP, WANDB, CONFIG))
    missing = set(sets) - SEEN
    if missing:
        raise SystemExit(f"not found in the notebook: {sorted(missing)}")
    cells = [
        cell("markdown", INTRO),
        cell("markdown", "## 1. Cài đặt (repo chính + repo này)"),
        cell("code", SETUP),
        cell("code", WANDB),
        cell("markdown", "## 2. Chọn phiên, dataset, checkpoint"),
        cell("code", CONFIG),
        cell("markdown", "## 3. Tải encoder / weight cần cho phiên"),
        cell("code", ASSETS),
        cell("markdown", "## 4. Chạy"),
        cell("code", RUN),
        cell("markdown", "## 5. Kết quả (kiến trúc, thông số, bảng, biểu đồ)"),
        cell("code", SHOW),
        cell("markdown", "## 6. Dọn Output"),
        cell("code", CLEAN),
    ]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"  # nbformat 4.5 requires cell ids
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
