"""Feature extraction with the exact encoders of VN-AV-DF (vn_av_df.training.prepare.make).

Cache layout is the training layout, <work>/<dataset>/cache/<encoder>/<sample_id>.npz (+ .json),
so `vn_av_df.training.data.BenchData` loads it unchanged. Encoders check their own signature and
skip files that are already valid, so an interrupted session simply resumes.

Unlike training (which stops on any residual error), a video that fails here (no face found,
no audio stream, ...) is recorded in failures-<encoder>.json and left out of scoring; the report
gives the failure count per dataset and class, because silently dropping hard videos would
inflate the AUC.
"""

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vn_av_df.common.runtime import write_json


def encoders_needed(model_names, extra=()):
    from vn_av_df.training.prepare import ORDER, encoders_for

    names = [m for m in model_names if m not in ("bl_ens_av", "avh_align_official")]
    if "avh_align_official" in model_names:
        names.append("avh_align")
    if "bl_ens_av" in model_names:
        names += ["avh_align_sup", "xlsr_frozen"]
    wanted = set(encoders_for(sorted(set(names)))) | set(extra)
    return [e for e in ORDER if e in wanted]


def extract(cfg, rows, clips, encoders, devices=(None,), run=None, log_every=25):
    """Run each encoder over `rows` (clips: {sample_id: normalized video}). Returns a report."""
    from vn_av_df.training.data import cache_file
    from vn_av_df.training.prepare import landmark_file, make

    cache = Path(cfg["cache"] if isinstance(cfg["cache"], str) else cfg["cache"][0])
    report = {}
    for name in encoders:
        spec = cfg["encoders"][name]
        if spec.get("cpu_only"):
            pool = [None] * int(spec.get("cpu_workers") or os.cpu_count() or 1)
        else:
            pool = [d for d in devices for _ in range(int(spec.get("workers_per_device", 1)))]
        folder = cache / name
        folder.mkdir(parents=True, exist_ok=True)
        todo = [r for r in rows if r["sample_id"] in clips]
        began = time.perf_counter()
        available = queue.Queue()
        for device in pool:
            encoder = make(cfg, name, device)
            getattr(encoder, "start", lambda: None)()
            available.put(encoder)
        guard, done, failed = threading.Lock(), [0], {}

        def one(row, encoder):
            sid = row["sample_id"]
            request = {"video": str(clips[sid])}
            if name in ("mouth", "lipinc"):
                request["box_cache"] = str(landmark_file(cfg, sid))
            elif name == "avhubert" and cache_file(cfg, "landmarks", sid).is_file():
                request["landmarks"] = str(cache_file(cfg, "landmarks", sid))
            encoder.extract(request, folder / f"{sid}.npz")

        def work(row):
            encoder = available.get()
            try:
                one(row, encoder)
            except Exception as exc:
                with guard:
                    failed[row["sample_id"]] = f"{type(exc).__name__}: {str(exc)[:300]}"
            finally:
                available.put(encoder)
            with guard:
                done[0] += 1
                if run is not None and (done[0] % log_every == 0 or done[0] == len(todo)):
                    spent = time.perf_counter() - began
                    run.log({f"extract/{name}/done": done[0],
                             f"extract/{name}/samples_per_min": done[0] / max(spent, 1e-6) * 60,
                             f"extract/{name}/failed": len(failed)})

        try:
            # A sample needs the previous encoders (landmarks before avhubert / mouth): skip rows
            # whose prerequisite failed instead of failing twice.
            ready = [r for r in todo if not any(r["sample_id"] in report.get(p, {}).get("failed", {})
                                                for p in report)]
            with ThreadPoolExecutor(max_workers=len(pool)) as executor:
                list(executor.map(work, ready))
        finally:
            while not available.empty():
                getattr(available.get(), "close", lambda: None)()
        elapsed = time.perf_counter() - began
        report[name] = {"samples": len(todo), "failed": failed, "elapsed_s": round(elapsed, 1)}
        write_json(folder / "failures.json", failed)
        print(f"== {name}: {len(todo) - len(failed)}/{len(todo)} ok in {elapsed / 60:.1f} min", flush=True)
    return report


def available(cfg, rows, encoders):
    """sample_ids that have a cache file for every encoder in `encoders`."""
    from vn_av_df.training.data import cache_index

    ids = [r["sample_id"] for r in rows]
    have = set(ids)
    for name in encoders:
        files = cache_index(cfg, name, ids)
        have &= {sid for sid, path in files.items() if Path(path).is_file()}
    return have
