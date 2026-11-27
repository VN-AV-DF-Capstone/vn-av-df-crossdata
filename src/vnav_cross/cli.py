"""python -m vnav_cross <step> [options]

Steps: manifest, prepare, extract, score, evaluate, report, run (all of the above), audit.
"""

import argparse
import json
from pathlib import Path

from vnav_cross import pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vnav_cross", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["manifest", "prepare", "extract", "score", "evaluate", "report", "run", "audit"])
    parser.add_argument("--config", help="YAML config (default: configs/crossdata.yaml)")
    parser.add_argument("--work", help="work folder (overrides the config)")
    parser.add_argument("--main-root", help="vn-av-df-forensics checkout (settings.py, weights)")
    parser.add_argument("--datasets", nargs="*", help="only these dataset keys")
    parser.add_argument("--session", default="all", choices=["all", "cpu", "gpu"], help="extract: encoder group")
    parser.add_argument("--device", help="cuda / cpu (default: cuda when available)")
    parser.add_argument("--workers", type=int, default=2, help="loader / ffmpeg workers")
    parser.add_argument("--wandb", action="store_true", help="log to W&B (needs WANDB_API_KEY)")
    parser.add_argument("--manifests", nargs="*", help="audit: VN-AV-DF manifest.jsonl files")
    parser.add_argument("--test-logs", nargs="*", default=[], help="audit: test-log.json files")
    parser.add_argument("--shortcut-audits", nargs="*", default=[], help="audit: shortcut_audit.json files")
    args = parser.parse_args(argv)

    cross = pipeline.load_config(args.config, work=args.work, main_root=args.main_root)
    if args.wandb:
        cross["wandb"] = {"mode": "online"}
    only = args.datasets or None
    steps = ["manifest", "prepare", "extract", "score", "evaluate", "report"] if args.step == "run" else [args.step]
    for step in steps:
        if step == "manifest":
            pipeline.step_manifest(cross, only)
        elif step == "prepare":
            pipeline.step_prepare(cross, only, args.workers)
        elif step == "extract":
            devices = [args.device] if args.device else None
            pipeline.step_extract(cross, only, args.session, devices)
        elif step == "score":
            pipeline.step_score(cross, only, args.device, args.workers)
        elif step == "evaluate":
            from vnav_cross.evaluate import step_evaluate

            step_evaluate(cross, only)
        elif step == "report":
            from vnav_cross.report import step_report

            step_report(cross)
        elif step == "audit":
            from vnav_cross.audit import run_audit
            from vnav_cross.report import step_audit_report

            manifests = args.manifests or []
            shortcut = args.shortcut_audits or [str(Path(m).with_name("shortcut_audit.json")) for m in manifests]
            audit = run_audit(manifests, args.test_logs, shortcut)
            out = Path(cross["work"]) / "results"
            pipeline.write_json(out / "audit.json", audit)
            print(json.dumps(audit["verdict"], indent=1), flush=True)
            print(f"audit report: {step_audit_report(cross, audit, out)}", flush=True)
    return 0
