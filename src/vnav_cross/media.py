"""Per-video preparation: normalize like the demo / training data and measure leading silence.

- `normalize` is `vn_av_df.demo.media.normalize`: 25 fps CFR, short side <= 720, <= 60 s, H.264 +
  AAC. Every detector then sees the same frame rate and container as in VN-AV-DF.
- We do NOT shift the audio with SyncNet (the demo does): an external fake can be misaligned on
  purpose, and re-aligning it would remove the evidence the sync branches look for.
- The original short side is kept: the M3-R resolution gate decides on the input video, not on
  the normalized copy.
- Leading silence follows AVH-Align (Smeu et al., CVPR 2025): the first sample whose magnitude
  exceeds tau = 5e-4, rounded up to a whole frame (1 / 25 s).
"""

import math
from pathlib import Path

import numpy as np

FPS = 25


def leading_silence(wave, sample_rate=16000, tau=5e-4, fps=FPS):
    """Seconds of leading silence (0 if the clip starts with sound, full length if never louder)."""
    wave = np.abs(np.asarray(wave, np.float32).reshape(-1))
    loud = np.flatnonzero(wave > tau)
    seconds = (loud[0] if len(loud) else len(wave)) / sample_rate
    return math.ceil(round(seconds * fps, 6)) / fps


def prepare(row, clips, tau=5e-4):
    """Normalize one video → info dict (or {"error": ...}); skips work already done."""
    from vn_av_df.data.media import decode_audio, probe
    from vn_av_df.demo.media import normalize

    clip = Path(clips) / f"{row['sample_id']}.mp4"
    info = {"sample_id": row["sample_id"], "clip": str(clip)}
    raw = None
    try:
        path = row["path"]
        if row.get("archive") and not clip.is_file():  # video inside zip volumes: pull this one out
            from vnav_cross.archive import extract

            raw = extract(row["archive"], row["member"], Path(clips) / "raw" / f"{row['sample_id']}.mp4")
            path = raw
        source = probe(path) if not (row.get("archive") and clip.is_file()) else probe(clip)
        info.update(width=source["width"], height=source["height"],
                    short_side=min(source["width"], source["height"]),
                    source_duration_s=source["duration_s"])
        if not clip.is_file():
            tmp = clip.with_suffix(".tmp.mp4")
            normalize(path, tmp)
            tmp.replace(clip)
        info["duration_s"] = probe(clip)["duration_s"]
        info["lead_silence_s"] = leading_silence(decode_audio(clip, 16000), 16000, tau)
    except Exception as exc:  # no audio stream, broken file, ...: reported, never hidden
        info["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if raw is not None:
            Path(raw).unlink(missing_ok=True)
    return info


def prepare_isolated(row, clips, tau=5e-4, timeout=300):
    """media.prepare in a child process: PyAV decoding from several threads of one process can
    deadlock (FakeAVCeleb run, 28/11: progress stopped after ~50 videos while each video alone took
    < 1 s), and a stuck video is killed after `timeout` seconds instead of blocking the session."""
    import json
    import subprocess
    import sys

    try:
        out = subprocess.run([sys.executable, "-m", "vnav_cross.media", json.dumps(row), str(clips), str(tau)],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"sample_id": row["sample_id"], "error": f"timeout after {timeout} s"}
    lines = out.stdout.strip().splitlines()
    if out.returncode or not lines:
        return {"sample_id": row["sample_id"], "error": f"worker exit {out.returncode}: {out.stderr[-300:]}"}
    return json.loads(lines[-1])


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(prepare(json.loads(sys.argv[1]), sys.argv[2], float(sys.argv[3]))))
