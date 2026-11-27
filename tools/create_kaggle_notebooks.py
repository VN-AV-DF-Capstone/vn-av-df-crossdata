"""Create every Kaggle notebook of the cross-dataset run, configured but NOT run (quick save: no quota).

    python tools/create_kaggle_notebooks.py [--owner linhxm] [--only cpu gpu report audit] [--dry-run]

Per dataset: <prefix>-<dataset>-cpu (no accelerator) and <prefix>-<dataset>-gpu (T4 x2, the cpu
notebook's Output attached as input), then <prefix>-report (all gpu Outputs) and <prefix>-audit.
Each notebook has its inputs, accelerator, Internet and cell-2 settings (SESSION, ONLY,
PREVIOUS_WORK) filled in. Kaggle does not let the API attach secrets: open each notebook once,
enable WANDB_API_KEY (Add-ons -> Secrets), then Save Version -> Save & Run All.
Kaggle drops a notebook input that has no version yet, so notebooks are created in dependency order.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = "linhxm/vn-av-df-weights-2seeds"
DATASETS = {
    "polyglotfake": "debajyatidey/polyglotfake",
    "fakeavceleb": "aicontentdetections/fakeavceleb-v1-2",
    "lavdf": "elin75/localized-audio-visual-deepfake-dataset-lav-df",
    "avdeepfake1m": "aicontentdetections/av-deepfake1m",
}
VN_PARTS = ["linhxm/vn-av-df-data-part1-trim04-fake", "linhxm/vn-av-df-data-part2-fake",
            "linhxm/vn-av-df-data-part3-fake"]


def plan(owner, prefix):
    work = "/kaggle/input/notebooks/{owner}/{slug}/crossdata"
    out = []
    for name, source in DATASETS.items():
        cpu, gpu = f"{prefix}-{name}-cpu", f"{prefix}-{name}-gpu"
        out.append({"slug": cpu, "kind": "cpu", "gpu": False, "datasets": [source, WEIGHTS], "kernels": [],
                    "set": {"SESSION": '"cpu"', "ONLY": f'["{name}"]'}})
        out.append({"slug": gpu, "kind": "gpu", "gpu": True, "datasets": [source, WEIGHTS],
                    "kernels": [f"{owner}/{cpu}"],
                    "set": {"SESSION": '"gpu"', "ONLY": f'["{name}"]',
                            "PREVIOUS_WORK": f'[Path("{work.format(owner=owner, slug=cpu)}")]'}})
    gpus = [f"{prefix}-{n}-gpu" for n in DATASETS]
    out.append({"slug": f"{prefix}-report", "kind": "report", "gpu": False, "datasets": [],
                "kernels": [f"{owner}/{g}" for g in gpus],
                "set": {"SESSION": '"report"', "ONLY": "None",
                        "PREVIOUS_WORK": "[" + ", ".join(f'Path("{work.format(owner=owner, slug=g)}")' for g in gpus) + "]"}})
    out.append({"slug": f"{prefix}-audit", "kind": "audit", "gpu": False, "datasets": VN_PARTS, "kernels": [],
                "set": {"SESSION": '"audit"'}})
    return out


def notebook_text(sets):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nb.ipynb"
        args = [sys.executable, str(ROOT / "tools/make_notebook.py"), "--out", str(path)]
        for k, v in sets.items():
            args += ["--set", f"{k}={v}"]
        subprocess.run(args, check=True, capture_output=True, text=True)
        return path.read_text(encoding="utf-8")


def save(api, owner, spec, text):
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

    request = ApiSaveKernelRequest()
    request.slug = f"{owner}/{spec['slug']}"
    request.new_title = spec["slug"]
    request.text = text
    request.language = "python"
    request.kernel_type = "notebook"
    request.is_private = True
    request.enable_gpu = spec["gpu"]
    request.enable_tpu = False
    request.enable_internet = True
    request.dataset_data_sources = spec["datasets"]
    request.kernel_data_sources = spec["kernels"]
    request.competition_data_sources = []
    request.model_data_sources = []
    if spec["gpu"]:
        request.machine_shape = "NvidiaTeslaT4"
    request.kernel_execution_type = KernelExecutionType.QUICK_SAVE
    with api.build_kaggle_client() as kaggle:
        response = kaggle.kernels.kernels_api_client.save_kernel(request)
    return getattr(response, "url", None) or getattr(response, "ref", None), getattr(response, "error", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="linhxm")
    parser.add_argument("--prefix", default="vn-av-df-crossdata")
    parser.add_argument("--only", nargs="*", choices=["cpu", "gpu", "report", "audit"])
    parser.add_argument("--names", nargs="*", help="only these notebook slugs (e.g. vn-av-df-crossdata-lavdf-cpu)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = [s for s in plan(args.owner, args.prefix)
             if (not args.only or s["kind"] in args.only) and (not args.names or s["slug"] in args.names)]
    order = {"cpu": 0, "audit": 0, "gpu": 1, "report": 2}
    specs.sort(key=lambda s: order[s["kind"]])
    api = None
    if not args.dry_run:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    for spec in specs:
        text = notebook_text(spec["set"])
        line = f"{spec['slug']:42s} {'T4x2' if spec['gpu'] else 'CPU ':5s} inputs={spec['datasets'] + spec['kernels']}"
        if args.dry_run:
            print("DRY", line, json.dumps(spec["set"]))
            continue
        url, error = save(api, args.owner, spec, text)
        print("ERROR" if error else "saved", line, url or "", error or "", flush=True)
        # a quick save still opens a short batch session to render the notebook: wait for it, or the
        # next save hits "Maximum batch CPU session count"
        for _ in range(60):
            status = str(api.kernels_status(f"{args.owner}/{spec['slug']}").status)
            if "RUNNING" not in status and "QUEUED" not in status:
                break
            time.sleep(5)


if __name__ == "__main__":
    main()
