"""Dataset adapters: each public dataset becomes one list of rows in a shared schema.

Row schema (one video):
    sample_id         "<dataset>_<16 hex>" (stable hash of the path relative to the dataset root)
    dataset, path     dataset key, absolute video path
    label             1 = contains AI manipulation (any modality), 0 = pristine
    synthetic_visual  face / lips generated or swapped
    synthetic_audio   speech generated or cloned
    category          dataset-native class (RVRA / FVRA / RVFA / FVFA, real / visual_modified / ...)
    method            generator name when the metadata gives it, else "unknown" ("real" for reals)
    language          spoken language when known
    subject           identity used for the subject-level bootstrap and the calibration split
    subset            MAVOS-DD only: core / extra_model / extra_language / extra_both
    fake_intervals, visual_fake_intervals, audio_fake_intervals   seconds, when the dataset has them

The adapters only read metadata and list files; nothing is copied or uploaded.
"""

import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def make_id(dataset, relative):
    key = str(PurePosixPath(*Path(relative).parts))
    return f"{dataset}_{hashlib.sha1(key.encode()).hexdigest()[:16]}"


def row(dataset, root, relative, *, visual, audio, category, method="unknown", language="en",
        subject=None, intervals=None, **extra):
    root, relative = Path(root), Path(relative)
    if relative.is_absolute():
        relative = relative.relative_to(root)
    intervals = intervals or {}
    out = {
        "sample_id": make_id(dataset, relative),
        "dataset": dataset,
        "path": str(root / relative),
        "relative": str(PurePosixPath(*relative.parts)),
        "label": int(bool(visual or audio)),
        "synthetic_visual": bool(visual),
        "synthetic_audio": bool(audio),
        "category": category,
        "method": method,
        "language": language,
        "subject": subject or str(PurePosixPath(*relative.parent.parts)),
        "fake_intervals": intervals.get("all", []),
        "visual_fake_intervals": intervals.get("visual", []),
        "audio_fake_intervals": intervals.get("audio", []),
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------- FakeAVCeleb
FAVC_TYPES = {
    "RealVideo-RealAudio": ("RVRA", False, False),
    "FakeVideo-RealAudio": ("FVRA", True, False),
    "RealVideo-FakeAudio": ("RVFA", False, True),
    "FakeVideo-FakeAudio": ("FVFA", True, True),
}


def favc_methods(root):
    """meta_data.csv (column names vary between releases) → {(identity, file name): method}."""
    out = {}
    path = next((Path(root) / n for n in ("meta_data.csv", "metadata.csv") if (Path(root) / n).is_file()), None)
    if path is None:
        return out
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            file = Path(r.get("path") or r.get("filename") or "").name
            if file:
                method = r.get("method") or "unknown"
                out[(r.get("source") or "", file)] = method
                out.setdefault(("", file), method)
    return out


def fakeavceleb(root, **_):
    """<root>/<type>/<race>/<gender>/<id>/<file>.mp4; the identity folder is the person shown."""
    root = Path(root)
    methods = favc_methods(root)
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT):
        parts = path.relative_to(root).parts
        kind = next((t for t in parts if t in FAVC_TYPES), None)
        if kind is None:
            continue
        category, visual, audio = FAVC_TYPES[kind]
        ident = path.parent.name
        method = "real" if category == "RVRA" else (
            methods.get((ident, path.name)) or methods.get(("", path.name)) or "unknown"
        )
        start = parts.index(kind)
        rows.append(row("fakeavceleb", root, path, visual=visual, audio=audio, category=category,
                        method=method, subject=ident,
                        race=parts[start + 1] if len(parts) > start + 3 else None,
                        gender=parts[start + 2] if len(parts) > start + 3 else None))
    return rows


# ---------------------------------------------------------------- AV-Deepfake1M / LAV-DF (one SDK schema)
MODIFY = {
    "real": (False, False),
    "visual_modified": (True, False),
    "audio_modified": (False, True),
    "both_modified": (True, True),
}
KIND = {flags: name for name, flags in MODIFY.items()}


def segments(value):
    return [[float(a), float(b)] for a, b in (value or [])]


def zip_members(folder):
    """{relative path inside the split: member name} of the zip volumes in `folder` (AV-Deepfake1M on
    Kaggle ships val/ as val.zip.001 ... .020), or None when the folder holds plain files."""
    from vnav_cross.archive import open_split_zip

    first = sorted(Path(folder).glob("*.zip.001")) or sorted(Path(folder).glob("*.zip"))
    if not first:
        return None, None
    with open_split_zip(first[0]) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(VIDEO_EXT)]
    return first[0], names


def avdeepfake1m(root, split="val", **_):
    """<root>/<split>_metadata.json + <root>/<split>/<file>; subject = VoxCeleb2 identity (first folder).
    When <root>/<split>/ holds zip volumes instead of videos, rows carry `archive` + `member` and
    media.prepare extracts only the sampled videos."""
    root = Path(root)
    candidates = (root / f"{split}_metadata.json", root / split / f"{split}_metadata.json")
    meta = next((p for p in candidates if p.is_file()), None)
    if meta is None:
        raise FileNotFoundError(f"{split}_metadata.json not found under {root}")
    base = root / split if (root / split).is_dir() else root
    archive, members = zip_members(base)
    lookup = {}
    for name in members or []:  # metadata "file" is relative to the split; members may carry a prefix
        parts = Path(name).parts
        for k in range(len(parts)):
            lookup.setdefault(str(PurePosixPath(*parts[k:])), name)
    rows = []
    for m in json.loads(meta.read_text(encoding="utf-8")):
        kind = m.get("modify_type", "real")
        visual, audio = MODIFY[kind]
        file = Path(m["file"])
        method = "real" if kind == "real" else (m.get("audio_model") if audio and m.get("audio_model") else "unknown")
        extra = {}
        if archive is not None:
            member = lookup.get(str(PurePosixPath(*file.parts)))
            if member is None:
                continue
            extra = {"archive": str(archive), "member": member}
        rows.append(row("avdeepfake1m", base, file, visual=visual, audio=audio, category=kind,
                        method=method, subject=file.parts[0],
                        intervals={"all": segments(m.get("fake_segments")),
                                   "visual": segments(m.get("visual_fake_segments")),
                                   "audio": segments(m.get("audio_fake_segments"))}, **extra))
    return rows


def lavdf(root, split="test", **_):
    """<root>/metadata.json (BA-TFD schema: file, split, modify_video, modify_audio, fake_periods,
    original). The subject is the source video, so a real and its fakes form one cluster."""
    root = Path(root)
    meta = next((root / n for n in ("metadata.json", "metadata.min.json") if (root / n).is_file()), None)
    if meta is None:
        raise FileNotFoundError(f"metadata.json not found under {root}")
    rows = []
    for m in json.loads(meta.read_text(encoding="utf-8")):
        if split and m.get("split") != split:
            continue
        visual, audio = bool(m.get("modify_video")), bool(m.get("modify_audio"))
        kind = KIND[(visual, audio)]
        periods = segments(m.get("fake_periods"))
        rows.append(row("lavdf", root, Path(m["file"]), visual=visual, audio=audio, category=kind,
                        method="real" if kind == "real" else "lavdf",
                        subject=Path(m.get("original") or m["file"]).stem,
                        intervals={"all": periods, "visual": periods if visual else [],
                                   "audio": periods if audio else []}))
    return rows


# ---------------------------------------------------------------- MAVOS-DD
def youtube_id(name):
    """First 11-character token between "_" / "." separators (e.g. 02690.png_Po82BhllEjA_340_1.mp4).
    An id that itself contains "_" is missed and the caller falls back to the file stem."""
    return next((t for t in re.split(r"[_.]", name) if re.fullmatch(r"[A-Za-z0-9-]{11}", t)), None)


def mavos_records(root):
    """MAVOS-DD metadata: an exported metadata.jsonl / metadata.csv, else the HF Arrow table."""
    root = Path(root)
    if (root / "metadata.jsonl").is_file():
        lines = (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    if (root / "metadata.csv").is_file():
        with open(root / "metadata.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    from datasets import Dataset  # pip install datasets

    return list(Dataset.load_from_disk(str(root)))


def flag(value):
    return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes")


def mavos_part(record):
    key = (flag(record.get("open_set_model")), flag(record.get("open_set_language")))
    return {(False, False): "core", (True, False): "extra_model",
            (False, True): "extra_language", (True, True): "extra_both"}[key]


# Test sets of the MAVOS-DD paper (Table 2) as unions of the parts above.
MAVOS_TEST_SETS = {
    "in_domain": ("core",),
    "open_model": ("core", "extra_model"),
    "open_language": ("core", "extra_language"),
    "open_full": ("core", "extra_model", "extra_language", "extra_both"),
}


def mavosdd(root, subsets=("in_domain", "open_model", "open_language"), **_):
    """Test split only. MAVOS-DD keeps the real speech in every fake, so synthetic_audio is False.
    The subject is the YouTube id inside the file name when there is one (approximate identity)."""
    root = Path(root)
    keep = set().union(*(MAVOS_TEST_SETS[s] for s in subsets))
    rows = []
    for r in mavos_records(root):
        if str(r.get("split")) != "test":
            continue
        part = mavos_part(r)
        if part not in keep:
            continue
        fake = str(r.get("label")).strip().lower() in ("fake", "1", "true")
        file = Path(r["video_path"])
        rows.append(row("mavosdd", root, file, visual=fake, audio=False,
                        category="fake" if fake else "real",
                        method=(r.get("generative_method") or "unknown") if fake else "real",
                        language=r.get("language") or "unknown",
                        subject=youtube_id(file.name) or file.stem, subset=part))
    return rows


# ---------------------------------------------------------------- PolyGlotFake
def polyglot_meta(path, key="video"):
    """{file name: record} from a PolyGlotFake json (a dict with a list under `key`, or a list)."""
    if not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get(key) or next((v for v in data.values() if isinstance(v, list)), []) if isinstance(data, dict) else data
    return {r.get("filename") or r.get("video_name") or r.get("name"): r for r in records if isinstance(r, dict)}


def polyglotfake(root, **_):
    """PolyGlotFake (Hou et al., 2024): <root>/real/<lang>/*.mp4 and <root>/fake/to_<lang>/*.mp4, labels in
    json_file/. Every fake has a translated, cloned voice (XTTS, Bark, MicroTTS, Vall-E-X, Tacotron)
    and re-synced lips (VideoRetalking or Wav2Lip), so visual and audio are both synthetic. The subject
    is the source video (e.g. es_1), shared by a real clip and all of its translations."""
    root = Path(root)
    fake_meta = polyglot_meta(root / "json_file/fake_Json_file/all_fake_video.json")
    real_meta = polyglot_meta(root / "json_file/real_json_file/all_real_video.json")
    rows = []
    for path in sorted((root / "real").rglob("*")):
        if path.suffix.lower() not in VIDEO_EXT:
            continue
        m = real_meta.get(path.name, {})
        lang = m.get("lang") or m.get("language") or path.parent.name
        rows.append(row("polyglotfake", root, path, visual=False, audio=False, category="real", method="real",
                        language=lang, subject=path.stem, source_language=lang))
    for path in sorted((root / "fake").rglob("*")):
        if path.suffix.lower() not in VIDEO_EXT:
            continue
        m = fake_meta.get(path.name, {})
        raw = m.get("raw_video_name") or path.name.split("_to_")[0] + path.suffix
        target = m.get("target_lang") or path.parent.name.removeprefix("to_")
        tts = m.get("tts_technique") or path.stem.rsplit("_", 1)[-1]
        sync = m.get("sync_tech") or "unknown"
        rows.append(row("polyglotfake", root, path, visual=True, audio=True, category="fake",
                        method=f"{tts}+{sync}", language=target, subject=Path(raw).stem,
                        source_language=m.get("raw_lang") or Path(raw).stem.split("_")[0],
                        tts=tts, lipsync=sync))
    return rows


# ---------------------------------------------------------------- VN-AV-DF itself (sanity check)
def vnavdf(root, split="test", **_):
    """VN-AV-DF part datasets (<root>/**/manifest.jsonl, clips next to it). Scoring the in-domain test
    split through this pipeline (normalize, extract, window) must reproduce the published test AUC
    closely; a large gap means the pipeline, not the detector, changed something."""
    rows = []
    for manifest in sorted(Path(root).rglob("manifest.jsonl")):
        base = manifest.parent
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if split and m["split"] != split:
                continue
            generator = m.get("generator") if m.get("generator") not in (None, "none") else None
            method = "+".join(x for x in (generator, m.get("voice_method")) if x) or "real"
            rows.append(row("vnavdf", base, Path(m["video"]), visual=m["synthetic_visual"],
                            audio=m["synthetic_audio"], category=m["cell"], method=method, language="vi",
                            subject=m["speaker_id"], kind=m.get("kind"),
                            intervals={"all": m.get("fake_intervals") or [],
                                       "visual": m.get("visual_fake_intervals") or [],
                                       "audio": m.get("audio_fake_intervals") or []}))
    return rows


# ---------------------------------------------------------------- any folder
def folder(root, **_):
    """<root>/real/** and <root>/fake/** (binary labels only, no modality information)."""
    root = Path(root)
    rows = []
    for name, fake in (("real", False), ("fake", True)):
        for path in sorted(p for p in (root / name).rglob("*") if p.suffix.lower() in VIDEO_EXT):
            rows.append(row("folder", root, path, visual=fake, audio=False, category=name,
                            method=path.parent.name if fake else "real", language="unknown",
                            subject=path.stem))
    return rows


ADAPTERS = {
    "vnavdf": vnavdf,
    "fakeavceleb": fakeavceleb,
    "avdeepfake1m": avdeepfake1m,
    "lavdf": lavdf,
    "mavosdd": mavosdd,
    "polyglotfake": polyglotfake,
    "folder": folder,
}

DISPLAY = {
    "vnavdf": "VN-AV-DF (test, in-domain)",
    "fakeavceleb": "FakeAVCeleb",
    "avdeepfake1m": "AV-Deepfake1M (val)",
    "lavdf": "LAV-DF (test)",
    "mavosdd": "MAVOS-DD (test)",
    "polyglotfake": "PolyGlotFake",
    "folder": "Custom folder",
}
# Tasks each dataset can score: MAVOS-DD has no fake audio; the folder adapter has no modality labels.
TASKS = {
    "vnavdf": ("ai", "visual", "audio"),
    "fakeavceleb": ("ai", "visual", "audio"),
    "avdeepfake1m": ("ai", "visual", "audio"),
    "lavdf": ("ai", "visual", "audio"),
    "mavosdd": ("ai", "visual"),
    # every fake is audio + visual, so the three labels coincide; the visual / audio columns then show
    # which head of a detector catches the fakes
    "polyglotfake": ("ai", "visual", "audio"),
    "folder": ("ai",),
}


def index(name, spec):
    """All rows of one dataset (before sampling)."""
    options = {k: v for k, v in spec.items() if k not in ("root", "sample")}
    return ADAPTERS[name](spec["root"], **options)
